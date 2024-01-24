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
#ifndef __DEF_HPP__
#define __DEF_HPP__

#define _USE_MATH_DEFINES

#include <random>
#include <cmath>
#include <pthread.h> 
#include <string>

/* NOTE: SKETCH_SIZE and K_HOPS are compilation
 * constant defined using -D flags. */

/* The value of items in the histogram
 * will decay every DECAY updates. */
extern int DECAY;
/* The rate of the decay. */
extern float LAMBDA;
/* We see WINDOW node updates before
 * we record a new graph sketch (if
 * BATCH is not used to determine
 * update frequency). This variable
 * is used if USEWINDOW (a compilation
 * flag) is set.*/
extern int WINDOW;
/* Batch size. If WINDOW is not used,
 * we record a new graph sketch every
 * BATCH edges are processed. */
extern int BATCH;
/* Determine if we want to chunk the string,
 * and how big each chunk is. CHUNK_SIZE > 1 */
extern bool CHUNKIFY;
extern int CHUNK_SIZE;
/* Sketch file to write the sketch. */
extern FILE * SFP;
#ifdef VIZ
/* Histogram file path to write histogram values.
 * We write one histogram per file. This is for
 * visualization only since file I/O is costly. */
extern std::string HIST_FILE;
#endif

/* In a streaming setting, GraphChi does not allow dynamic vertex/edge type.
 * We therefore must fixed the neighborhood we are exploring.
 * Currently we implement K_HOPS neighborhood.
 * Therefore, The array "src" holds K_HOPS + 1 elements:
 * - The label of the source node, which we can initialize from the file.
 * - The relabel of the first-hop neighborhood.
 * - The relabel of the second-hop neighborhood.
 * - The relabel of the third-hop neighborhood.
 * - ...
 * - The relabel of the K_HOPS neighborhood.
 * The array "tme" is the corresponding timestamps. The first element is the timestamp of the edge.
 * The following elements are the corresponding timestamps of the relabeled node at each hop.
 * "dst" is the label of the destination node.
 * "itr" is the number of times the destination node has access the edge for computation.
 * "edg" is the edge label.
 * For streaming new edges:
 * - "new_src": whether the source node is new, never-before-seen.
 * - "new_dst": whether the destination node is new. never-before-seen.
 */
typedef struct edge_label {
    /* We use K_HOPS+1 because the first element is itself and
     * the next K_HOPS are exploration of K_HOPS neighbors. */
    unsigned long src[K_HOPS+1];
    unsigned long tme[K_HOPS+1];
    unsigned long dst;
    unsigned long edg;
    int itr;
    bool new_src;
    bool new_dst;
} EdgeDataType;

/* Node remembers all its most-updated labels "lb" and timestamps "tm".
 * "is_leaf" is whether the node is a leaf node (i.e., no incoming edges). */
typedef struct node_label {
    unsigned long lb[K_HOPS+1];
    unsigned long tm[K_HOPS+1];
    bool is_leaf;
} VertexDataType;

/* Each histogram element is associated with r, beta, c, which
 * are parameters to create hash values. r ~ Gamma(2, 1),
 * c ~ Gamma(2, 1), beta ~ Uniform(0, 1)
 * Reference: https://doc.rero.ch/record/309000/files/cud_hsf.pdf
 */
struct hist_elem {
    double r[SKETCH_SIZE];
    double beta[SKETCH_SIZE];
    double c[SKETCH_SIZE]; 
};
/* Distribution used in locality-sensitive hashing. */
std::gamma_distribution<double> gamma_dist(2.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

#endif /* __DEF_HPP__ */
