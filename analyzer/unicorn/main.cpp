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
#include <pthread.h>
#include <sys/types.h>
/* Unicorn header files we use. */
/* NOTE: helper.hpp must be included first.
 * It defines the parser that overwrites
 * GraphChi's dummy parser.*/
#include "include/helper.hpp"
#include "include/def.hpp"
#include "include/histogram.hpp"
#include "../extern/extern.hpp"
#include "wl.hpp"
/* GraphChi header files we use. */
#include "graphchi_basic_includes.hpp"
#include "logger/logger.hpp"

using namespace graphchi;

graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType> * dyngraph_engine;
std::string stream_file;
std::string sketch_file;
/* The following variables are declared
 * in extern.hpp. They are defined here
 * and will be used in various place in
 * GraphChi and Unicorn. */
pthread_barrier_t std::graph_barrier;
pthread_barrier_t std::stream_barrier;
int std::stop = 0;
bool std::base_graph_constructed = false;
bool std::no_new_tasks = false;
/* The following variables are declared in def.hpp.
 * They are defined here and will be assigned values
 * in the main function. */
int DECAY;
float LAMBDA;
int WINDOW;
int BATCH;
bool CHUNKIFY = true;
int CHUNK_SIZE;
FILE * SFP;
#ifdef VIZ
std::string HIST_FILE;
#endif

/*!
 * @brief A separate thread execute this function to stream graph from a file.
 */
void * dynamic_graph_reader(void * info) {
#ifdef DEBUG
    logstream(LOG_DEBUG) << "Stream provenance graph from file: " << stream_file << std::endl;
#endif
    /* A busy loop to wait until the base graph histogram is constructed. */
    while(!std::base_graph_constructed) {
#ifdef DEBUG
        logstream(LOG_DEBUG) << "Waiting for the base graph to be constructed..." << std::endl;
#endif
        sleep(0);
    }
    /* Once breaking out of the loop, we know 
     * the base graph histogram is ready.
     * Get the histogram map singleton. */
    Histogram* hist = Histogram::get_instance();
    /* Initailize the first sketch of the histogram. */
    hist->create_sketch();
    /* If BASESKETCH is set, we record the first sketch
     * from the base graph. BASESKETCH is recommended to
     * be set if USEWINDOW is also set. Do NOT set
     * BASESKETCH if USEWINDOW is not set. */
    /* SFP must exist to write. */
    if (SFP == NULL)
	logstream(LOG_ERROR) << "Sketch file no longer exists..." << std::endl;
    assert(SFP != NULL);
#ifdef BASESKETCH
    for (int i = 0; i < SKETCH_SIZE; i++)
	fprintf(SFP,"%lu ", hist->get_sketch()[i]);
    fprintf(SFP, "\n");
#endif
    /* Open the file for streaming. */
    FILE * f = fopen(stream_file.c_str(), "r");
    if (f == NULL)
        logstream(LOG_ERROR) << "Unable to open the file for streaming: " << stream_file << ". Error code: " << strerror(errno) << std::endl;
    assert(f != NULL);
    /* Read the file. */
    vid_t srcID;
    vid_t dstID;
    EdgeDataType e;
    /* Char buffer to hold each line. */
    char s[1024];
    /* Count the number of batched edges. */
    int cnt = 0;
    /* For synchronization with GraphChi algorithm. */
    bool passed_barrier = false;

    while(fgets(s, 1024, f) != NULL) {
        /* We add more edges for the GraphChi WL to compute, but we
         * will wait until all the previously added edges have finished
         * before we add new ones. */
        if (cnt == 0 && !passed_barrier) {
	    /* We are waiting at the stream_barrier here until
	     * GraphChi WL hits the same stream_barrier (when it
	     * finishes the current batch for all nodes), and
	     * then we will start streaming new edges. */
            pthread_barrier_wait(&std::stream_barrier);
	    /* If USEWINDOW is not set, we record a new sketch
	     * every BATCH streaming edges are processed. We
	     * also record the first sketch as the base graph
	     * automatically. */
#ifndef USEWINDOW
	    for (int i = 0; i < SKETCH_SIZE; i++)
		fprintf(SFP,"%lu ", hist->get_sketch()[i]);
	    fprintf(SFP, "\n");
#ifdef VIZ
	    /* We output a histogram file (one histogram per file)
	     * for visualization. */
	    hist->write_histogram();
#endif
#endif
        }
        passed_barrier = true;
	FIXLINE(s);
        /* Parse the line. */
        char delims[] = ":\t ";
        unsigned char *t;
        char *k;

	/* Obtain source node ID. */
        k = strtok(s, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Source ID is missing." << std::endl;
        assert(k != NULL);
        srcID = atoi(k);

	/* Obtain destination node ID. */
        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Destination ID is missing." << std::endl;
        assert(k != NULL);
        dstID = atoi(k);

	/* Populate EdgeDataType for the edge. */
        e.itr = 0; /* new itr count is always 0. */
        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Source label is missing." << std::endl;
        assert(t != NULL);
        e.src[0] = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Destination label is missing." << std::endl;
        assert (t != NULL);
        e.dst = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Edge label is missing." << std::endl;
        assert (t != NULL);
        e.edg = hash(t);

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "New_src info is missing." << std::endl;
        assert(k != NULL);
        int new_src = atoi(k);
        if (new_src == 1)
            e.new_src = true;
        else
            e.new_src = false;

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "New_dst info is missing." << std::endl;
        assert(k != NULL);
        int new_dst = atoi(k);
        if (new_dst == 1)
            e.new_dst = true;
        else
            e.new_dst = false;

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Time is missing." << std::endl;
        assert (k != NULL);
        e.tme[0] = strtoul(k, NULL, 10);

#ifdef DEBUG
        k = strtok(NULL, delims);
        if (k != NULL)
            logstream(LOG_DEBUG) << "Extra info in the edge is ignored." << std::endl;
#endif
        if (srcID == dstID) {
#ifdef DEBUG
            logstream(LOG_ERROR) << "Ignore an edge because it is a self-loop: " << srcID << "<->" << dstID <<std::endl;
#endif
            continue;
        }
        /* Add the new edge to the graph. */
        bool success = false;
	/* Try to add until it is successful. */
        while (!success)
            success = dyngraph_engine->add_edge(srcID, dstID, e);
        ++cnt;
        /* Schedule the new nodes to be computed. 
         * TODO: probably not needed since we are
	 *       not doing selective scheduling. */
        dyngraph_engine->add_task(srcID);
        dyngraph_engine->add_task(dstID);
#ifdef DEBUG
        logstream(LOG_DEBUG) << "Schedule a new edge: " << srcID << " -> " << dstID << std::endl;
#endif
        if (cnt == BATCH) {
            /* We continue to add new edges until INTERVAL edges are added.
	     * When we are in this block, we have added INTERVAL edges and
	     * we can now let GraphChi WL starts its computation. */
            cnt = 0;
            passed_barrier = false;
	    /* GraphChi WL would be waiting for us to hit this
	     * graph_barrier barrier. Once we hit this barrier,
	     * GraphChi WL will resume its execution on our
	     * newly added nodes and edges. */
            pthread_barrier_wait(&std::graph_barrier);
        }
    }
    /* Signal to GraphChi WL that we have streamed all the edges. 
     * So when GraphChi WL finishes computation, it will stop. */
    std::stop = 1;
    if (cnt != 0) {
	    /* This block handles leftover edges that do not
	     * hit INTERVAL in the previous loop. We still
	     * want GraphChi WL to process them. */
	    pthread_barrier_wait(&std::graph_barrier);
    }

    /* We are done. Close the stream file. */
    if (ferror(f) != 0 || fclose(f) != 0) {
        logstream(LOG_ERROR) << "Unable to close the stream file: " << stream_file << ". Error code: " << strerror(errno) << std::endl;
	return NULL;
    }
    
    return NULL;
}

/* Run the program using command line on the graphchi-cpp directory:
 * bin/streaming/main file streaming/test.data niters 1000 stream_file streaming/stream.data
 * Compile the program:
 * With debugging information: make sdebug
 * Without debugging info: make streaming/main
 */
int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line arguments and the configuration file. */
    graphchi_init(argc, argv);

    /* Metrics object for keeping track of performance
     * counters and other information. Currently required. */
    metrics m("Streaming Extractor");

    /* Set up logging. */
#ifdef DEBUG
    global_logger().set_log_level(LOG_DEBUG);
#else
    global_logger().set_log_level(LOG_INFO);
#endif

    /* Parameters from command line. */
    /* The base graph file path. */
    std::string base_file = get_option_string("base");
    /* Number of iterations (default: 1000000). 
     * This is a upper limit. We don't
     * actually run that many iterations.
     * GraphChi WL stops when there is no more
     * new nodes or edges to process. */
    int niters = get_option_int("niters", 1000000);
    bool scheduler = true;	/* We always use GraphChi scheduler. */
    stream_file = get_option_string("stream");
    /* More parameters from the command line to configure
     * hyperparameters of feature vector generation. Those
     * variables are declared extern somewhere else. */
    DECAY = get_option_int("decay", 10);
    LAMBDA = get_option_float("lambda", 0.02);	
    BATCH = get_option_int("batch", 1000);
    WINDOW = get_option_int("window", 500);
    sketch_file = get_option_string("sketch");
#ifdef VIZ
    HIST_FILE = get_option_string("histogram");
#endif
    int to_chunk = get_option_int("chunkify", 1);
    if (!to_chunk) CHUNKIFY = false;
    CHUNK_SIZE = get_option_int("chunk_size", 5);

    /* Open the sketch file to write. */
    SFP = fopen(sketch_file.c_str(), "a");
    if (SFP == NULL) {
        logstream(LOG_ERROR) << "Cannot open the sketch file to write: " << sketch_file << ". Error code: " << strerror(errno) << std::endl;
    }
    assert(SFP != NULL);

    /* Process input file - if not already preprocessed */
    int nshards = convert_if_notexists<EdgeDataType>(base_file, get_option_string("nshards", "auto"));

    /* Create the engine object. */
    dyngraph_engine = new graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType>(base_file, nshards, scheduler, m); 

    /* Initialize barrier. */
    pthread_barrier_init(&std::stream_barrier, NULL, 2);
    pthread_barrier_init(&std::graph_barrier, NULL, 2);

    /* Start streaming thread. */
    pthread_t strthread;
    int ret = pthread_create(&strthread, NULL, dynamic_graph_reader, NULL);
    assert(ret >= 0);

    /* Run the engine */
    WeisfeilerLehman program;
    dyngraph_engine->run(program, niters);

    /* Once streaming is done, we will record the last 
     * sketch that describes the entire graph. */
    /* We append the last sketch to the sketch file. */
    Histogram* hist = Histogram::get_instance();
#ifdef DEBUG
    logstream(LOG_DEBUG) << "Recording the final graph sketch..." << std::endl;
#endif
    if (SFP == NULL)
        logstream(LOG_ERROR) << "Sketch file no longer exists..." << std::endl;
    assert(SFP != NULL);
    hist->record_sketch(SFP);
    /* Once we are done, we close the sketch file. */
    if (ferror(SFP) != 0 || fclose(SFP) != 0) {
        logstream(LOG_ERROR) << "Unable to close the sketch file: " << sketch_file <<  std::endl;
        return -1;
    }

    /* Release the barrier resources. */
    int ret_stream = pthread_barrier_destroy(&std::stream_barrier);
    int ret_graph = pthread_barrier_destroy(&std::graph_barrier);
    if (ret_stream == EBUSY) {
        logstream(LOG_ERROR) << "Resource stream_barrier cannot be destroyed." << std::endl;
    }
    if (ret_graph == EBUSY) {
        logstream(LOG_ERROR) << "Resource graph_barrier cannot be destroyed." << std::endl;
    }
#ifdef DEBUG
    metrics_report(m);
#endif
    return 0;
}
