/*
 *
 * Author: Xueyuan Han <hanx@g.harvard.edu>
 *
 * Copyright (C) 2018-2020 Harvard University
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2, as
 * published by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 */

#include <fstream>
#include <math.h>
#include <random>
#include <cstdlib>
#include <string>

#include "include/histogram.hpp"

Histogram* Histogram::histogram;

/* Singleton should always return the same instance. */
Histogram* Histogram::get_instance() {
    if (!histogram)
        histogram = new Histogram();
    return histogram;
}

Histogram::~Histogram() {
    delete histogram;
}

/* Sample random values for hashing histogram. */
struct hist_elem Histogram::construct_hist_elem(unsigned long label) {
    struct hist_elem new_elem;
    std::default_random_engine r_generator(label);
    std::default_random_engine c_generator(label / 2);
    std::default_random_engine beta_generator(label);
    for (int i = 0; i < SKETCH_SIZE; i++) {
        new_elem.r[i] = gamma_dist(r_generator);
        new_elem.beta[i] = uniform_dist(beta_generator);
        new_elem.c[i] = gamma_dist(c_generator);
    }
    gamma_dist.reset();
    return new_elem;
}

/* Decay values in the histogram map, and record the sketch to the 
 * file @fp, if WINDOW updates have performed (if WINDOW is used). */
void Histogram::decay(FILE* fp) {
    this->histogram_map_lock.lock();
    this->t++;
#ifdef USEWINDOW
    this->w++;
#endif
    /* Decay only when t == DECAY. */
    if (this->t >= DECAY) {
        std::map<unsigned long, double>::iterator it;
	/* Decay histogram values. */
        for (it = this->histogram_map.begin(); it != this->histogram_map.end(); it++)
            it->second *= this->powerful;
	/* Decay sketch values. */
        for (int i = 0; i < SKETCH_SIZE; i++)
            this->hash[i] *= this->powerful;
        this->t = 0;  /* Reset the timer. */
    }
    /* Record sketch only when t == WINDOW if we use
     * WINDOW as frequency to generate sketches. */
#ifdef USEWINDOW
    if (this->w >= WINDOW) {
        for (int i = 0; i < SKETCH_SIZE; i++)
            fprintf(fp,"%lu ", this->sketch[i]);
        fprintf(fp, "\n");
        this->w = 0; /* Reset the timer. */
#ifdef VIZ
	/* If we write sketch to a file, we will also Write its
	 * corresponding histogram to a separate file too. We
	 * write one histogram per file (but we write all sketches
	 * to one file). */
	this->write_histogram();
#endif
    }
#endif
    this->histogram_map_lock.unlock();
}


/* Insert @label to the histogram if it does not exist; otherwise, update its value.
 * If @base true, we do not update hash value; we only update them during streaming.
 * We do not decay the histogram or the sketch in this function. */
void Histogram::update(unsigned long label, bool base) {
    this->histogram_map_lock.lock();
    /* We add the new element or update the existing element in the
     * histogram. This is done both in base and stream graph. */
    std::pair<std::map<unsigned long, double>::iterator, bool> rst;
    double counter = 1;
    rst = this->histogram_map.insert(std::pair<unsigned long, double>(label, counter));
    if (rst.second == false) {
#ifdef DEBUG
        logstream(LOG_DEBUG) << "The label " << label << " is already in the map. Updating the sketch and its hash..." << std::endl;
#endif
        (rst.first)->second++;
    }
    /* Now we update the hash if needed.
     * Update hash only in stream graph. */
    if (!base) {
        /* MEMORY is a compilation constant defined using -D flag.
	 * If MEMORY is set, we use pre-sampled random variable.
	 * This is an optimization as sampling can be slow on-the-fly. */
#ifdef MEMORY
	/* We use @label to decide which pre-sampled values to use
	 * so we will always use the same values for the same label. */
	srand(label);
	int pos1 = rand() % PREGEN;
	int pos2 = rand() % PREGEN;
	for (int i = 0; i < SKETCH_SIZE; i++) {
            /* Compute the new hash value using picked random variables. */
            double c = this->gamma_param[pos2][i];
	    double y = (rst.first)->second / this->r_beta_param[pos1][i];
	    double a = c / (y * this->power_r[pos1][i]);
            /* If the hash value is smaller than the existing value,
	     * we replace the hash value and change the sketch value. */
	    if (a < this->hash[i]) {
                this->hash[i] = a;
		this->sketch[i] = (rst.first)->first;
	    }
	}
#else
        /* If we do not use pre-sampled random values, we sample first
	 * using the label and then update the hash values and sketches. */
        struct hist_elem generated_param = this->construct_hist_elem(label);
	for (int i = 0; i < SKETCH_SIZE; i++) {
            /* Compute the new hash value a. */
            double r = generated_param.r[i];
	    double beta = generated_param.beta[i];
	    double c = generated_param.c[i];
	    double y = pow(M_E, log((rst.first)->second) - r * beta);
	    double a = c / (y * pow(M_E, r));

	    if (a < this->hash[i]) {
                this->hash[i] = a;
		this->sketch[i] = (rst.first)->first;
            }
	}
#endif
    }
    this->histogram_map_lock.unlock();
    return;
}

/* Create (and initialize) a sketch after the base graph has been processed by GraphChi WL.
 * This function is called only once during initialization. If MEMORY is set to 1, we also
 * pre-sample some random values to speed up computations later. */
void Histogram::create_sketch() {
    this->histogram_map_lock.lock();
#ifndef MEMORY
    /* If we decide not to pre-sample, we can still optimize a bit by
     * locally saving some sketch parameters for to initialize sketched. */
    std::map<unsigned long, struct hist_elem> base_map;
    for (std::map<unsigned long, double>::iterator it = this->histogram_map.begin(); it != this->histogram_map.end(); it++) {
        unsigned long label = it->first;
	struct hist_elem new_elem = this->construct_hist_elem(label);
	base_map.insert(std::pair<unsigned long, struct hist_elem>(label, new_elem));
    }

    for (int i = 0; i < SKETCH_SIZE; i++) {
        /* Compute the hash value. */
        std::map<unsigned long, double>::iterator histoit = this->histogram_map.begin();
	unsigned long label = histoit->first;
	/* First find saved local random values. */
	std::map<unsigned long, struct hist_elem>::iterator basemapit;
	basemapit = base_map.find(label);
	if (basemapit == base_map.end()){
            logstream(LOG_ERROR) << "Label: " << label << " should exist in local base map, but it does not. " << std::endl;
	    assert(false);
        }
	struct hist_elem histo_param = basemapit->second;
	/* We use those values to compute hash. */
	double y = pow(M_E, log(histoit->second) - histo_param.r[i] * histo_param.beta[i]);
	double a_i = histo_param.c[i] / (y * pow(M_E, histo_param.r[i]));
	unsigned long s_i = histoit->first;
	for (histoit = this->histogram_map.begin(); histoit != this->histogram_map.end(); histoit++) {
            label = histoit->first;
	    basemapit = base_map.find(label);
	    if (basemapit == base_map.end()){
                logstream(LOG_ERROR) << "Label: " << label << " should exist in local base map, but it does not. " << std::endl;
                assert(false);
            }
            histo_param = basemapit->second;
	    y = pow(M_E, log(histoit->second) - histo_param.r[i] * histo_param.beta[i]);
	    double a = histo_param.c[i] / (y * pow(M_E, histo_param.r[i])); 
	    if (a < a_i) {
                a_i = a;
		s_i = histoit->first;
	    }
	}
	this->sketch[i] = s_i;
	this->hash[i] = a_i;
    }
#else
    /* Pre-sample random variables. */
    srand(36); /* Set a seed. */
    for (unsigned long i = 0; i < (unsigned long)PREGEN; i++) {
        int randomized_i = rand();
	std::default_random_engine r_generator(randomized_i);
	std::default_random_engine beta_generator(randomized_i);

	for (int j = 0; j < SKETCH_SIZE; j++) {
            this->gamma_param[i][j] = gamma_dist(r_generator);
	    double uniform_param = uniform_dist(beta_generator);
	    this->r_beta_param[i][j] = pow(M_E, this->gamma_param[i][j] * uniform_param);
	    this->power_r[i][j] = pow(M_E, this->gamma_param[i][j]);
	}
	gamma_dist.reset();
    }
    /* Initialize sketch. */
    for (int i = 0; i < SKETCH_SIZE; i++) {
        std::map<unsigned long, double>::iterator histoit = this->histogram_map.begin();
	unsigned long label = histoit->first;

	srand(label);
	int pos1 = rand() % PREGEN; /* For gamma and uniform distribution. */
	int pos2 = rand() % PREGEN; /* For the other gamma distribution. */

	double y = histoit->second / this->r_beta_param[pos1][i];
	double a_i = this->gamma_param[pos2][i] / (y * this->power_r[pos1][i]);
	unsigned long s_i = histoit->first;
	for (histoit = this->histogram_map.begin(); histoit != this->histogram_map.end(); histoit++) {
            label = histoit->first;

	    srand(label);
	    pos1 = rand() % PREGEN;
	    pos2 = rand() % PREGEN;

	    y = histoit->second / this->r_beta_param[pos1][i];
	    double a = this->gamma_param[pos2][i] / (y * this->power_r[pos1][i]);
	    if (a < a_i) {
                a_i = a;
		s_i = histoit->first;
	    }
	}
	this->sketch[i] = s_i;
	this->hash[i] = a_i;
    }
#endif
    this->histogram_map_lock.unlock();
    return;
}

/* Write the sketch to the file @fp. */
void Histogram::record_sketch(FILE* fp) {
    this->histogram_map_lock.lock();
    for (int i = 0; i < SKETCH_SIZE; i++) {
        fprintf(fp,"%lu ", this->sketch[i]);
    }
    fprintf(fp, "\n");
    this->histogram_map_lock.unlock();
    return;
}

/* Getter: to get the sketch. */
unsigned long* Histogram::get_sketch() {
    return this->sketch;
}

#ifdef VIZ
/* Write the histogram to a file. */
void Histogram::write_histogram() {
    std::string hist_file_name(HIST_FILE); /* HIST_FILE is defined in def.hpp */
    hist_file_name += ".";
    hist_file_name += std::to_string(this->c);
    /* Open the file to write. */
    FILE *hfp = fopen(hist_file_name.c_str(), "w");
    if (hfp == NULL)
        logstream(LOG_ERROR) << "Cannot open the histogram file to write: " << hist_file_name << ". Error code: " << strerror(errno) << std::endl;
    assert(hfp != NULL);
    /* Write to the histogram file. */
    std::map<unsigned long, double>::iterator it;
    for (it = this->histogram_map.begin(); it != this->histogram_map.end(); it++)
        fprintf(hfp,"%lu,%lf\n", it->first, it->second);
    fprintf(hfp, "\n");
    /* Close the file */
    if (ferror(hfp) != 0 || fclose(hfp) != 0) {
        logstream(LOG_ERROR) << "Unable to close the histogram file: " << hist_file_name << std::endl;
        assert(false);
    }
    /* Use this->c to create different file names for different histograms. */
    this->c++;
    return;
}
#endif

#ifdef DEBUG
/* Print the histogram map for debugging. */
void Histogram::print_histogram() {
    std::map<unsigned long, double>::iterator it;
    logstream(LOG_DEBUG) << "Printing the histogram for debugging..." << std::endl;
    for (it = this->histogram_map.begin(); it != this->histogram_map.end(); it++)
        logstream(LOG_DEBUG) << "[" << it->first << "]->" << it->second << "  ";
    logstream(LOG_DEBUG) << "\n";
    return;
}
#endif
