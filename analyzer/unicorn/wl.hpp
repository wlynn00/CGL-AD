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
#include <thread>
#include <mutex>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <pthread.h>
/* GraphChi header files. */
#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "logger/logger.hpp"
/* Unicorn header files. */
#include "../extern/extern.hpp"
#include "include/def.hpp"
#include "include/helper.hpp"
#include "include/histogram.hpp"

namespace graphchi {
    /* GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
     * class. The main logic is usually in the update function. */
    struct WeisfeilerLehman : public GraphChiProgram<VertexDataType, EdgeDataType> {
        /* Get the histogram singleton. */
        Histogram* hist = Histogram::get_instance();

        /* Vertex update function. */
        void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
            /* Detected isolated vertex in the graph. The following code 
	     * runs only during debugging to discover dirty data. */
#ifdef DEBUG
            if (vertex.num_edges() <= 0) {
	        logstream(LOG_DEBUG) << "Isolated vertex #"<< vertex.id() <<" detected." << std::endl;
		assert(false);
	    }
#endif
            if (gcontext.iteration == 0) {
	        /* On the first iteration, initialize vertex label
		 * on the base graph (before new edges stream in). */
		VertexDataType nl;

		if (vertex.num_inedges() > 0) {
		    graphchi_edge<EdgeDataType> * edge = vertex.inedge(0); /* Use the first inedge to get its original label. */
		    nl.lb[0] = edge->get_data().dst;
		    nl.is_leaf = false;

		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			el.itr++; /* After this initialization, every edge in the base graph has "itr" value 1. */
			in_edge->set_data(el);
		    }
		} else {
		    /* If it does not have any incoming neighbors. Then
		     * it must have at least one out-going neighbor. */
                    graphchi_edge<EdgeDataType> * edge = vertex.random_outedge();
                    nl.lb[0] = edge->get_data().src[0];
                    nl.is_leaf = true;
		}
		nl.tm[0] = 0; /* The first timestamp associated with a vertex is always zero. */
		vertex.set_data(nl);

		/* Populate the histogram. */
		hist->update(nl.lb[0], true);

		/* Schedule itself for the next iteration. */
		if (gcontext.scheduler != NULL) {
		    gcontext.scheduler->add_task(vertex.id());
		}
#ifdef DEBUG
		logstream(LOG_DEBUG) << "Original Label (" << vertex.id() << "): " << nl.lb[0] << std::endl;
#endif
            } else if (gcontext.iteration < K_HOPS + 1){	/* we know after K_HOPS iterations, we will be done with the base graph. */
                /* After the first iteration, all nodes in the base graph are initialized. 
                 * All edges in the base graph should have "itr" >= 1. */
#ifdef DEBUG
		/* This is simply a check to make sure that every vertex in the graph
		 * at this point belongs to the base graph. */
		for (int i = 0; i < vertex.num_outedges(); i++) {
		    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
		    EdgeDataType el = out_edge->get_data();
		    if (el.itr == 0)
			assert(false);
		}
#endif
		/* We need to finish iterating the base graph before we handle new edges.
		 * Note that while we are iterating the base graph, new edges or nodes 
		 * will not be added to the graph. If CHUNKIFY is set, we will also segment
		 * the concatenated string. That is, we may add multiple entries to the map 
		 * for one string. */
                std::vector<EdgeDataType> neighborhood; /* We reuse edge_label struct vector to store the neighborhood values. */
		for (int i = 0; i < vertex.num_inedges(); i++) {
                    graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
		    EdgeDataType el = in_edge->get_data();
		    assert(el.itr == gcontext.iteration);	/* During base graph iteration, edge itr value should be the same as gcontext iteration value before the update. */
		    neighborhood.push_back(el);
		    /* We will use those edges so increment the itr count by 1 and update the edge. */
		    el.itr++;
		    in_edge->set_data(el);
		}
		VertexDataType nl = vertex.get_data();

		if (neighborhood.size() == 0) {
                    /* The vertex could also be a node in the base graph that
		     * does not have any in-coming edges, i.e., a vertex with
		     * is_leaf == true.  Simply use the last label of the vertex
		     * itself since it has no incoming neighbors. */
		    unsigned long last_itr_label = nl.lb[gcontext.iteration - 1];
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "The label string of the base leaf vertex (" << vertex.id() << "): " << last_itr_label << std::endl;
#endif
		    /* Populate the histogram. */
		    hist->update(last_itr_label, true);
		    /* Update the vertex's label vector. */
		    nl.lb[gcontext.iteration] = last_itr_label;
		    nl.tm[gcontext.iteration] = 0; /* All timestamps of the leaf vertex is set to be 0. */
		    vertex.set_data(nl);
		    /* Update its out-going edges.*/
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[gcontext.iteration] = last_itr_label;
			/* Time stamp does not change for nodes with no in-coming neighbors. */
			el.tme[gcontext.iteration] = el.tme[gcontext.iteration - 1]; 
			out_edge->set_data(el);
		    }
		} else {
		    /* We first sort the labels based on the timestamps of the in_edges.
		     * Note that the neighborhood only contains edges of the base graph. */
		    std::sort(neighborhood.begin(), neighborhood.end(), EdgeSorter(gcontext.iteration - 1));
		    /* First construct the string of the vertex itself. */
		    std::string new_label_str = "";
		    std::string first_str;
		    std::stringstream first_out;
		    first_out << vertex.get_data().lb[gcontext.iteration - 1];
		    first_str = first_out.str();
		    new_label_str += first_str; /* Use space to separate number strings. */
		    /* Then append neighborhood labels. */
		    for (std::vector<EdgeDataType>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
			if (gcontext.iteration == 1) { /* The first iteration includes edge labels. */
			    std::string edge_str;
			    std::stringstream edge_out;
			    edge_out << it->edg;
			    edge_str = edge_out.str();
			    new_label_str += " " + edge_str;
			}
			std::string node_str;
			std::stringstream node_out;
			node_out << it->src[gcontext.iteration - 1];
			node_str = node_out.str();
			new_label_str += " " + node_str;
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label string of vertex (" << vertex.id() << "): " << new_label_str << std::endl;
#endif
		    /* Relabel by hashing. */
		    unsigned long new_label = hash((unsigned char *)new_label_str.c_str());
		    /* Populate the histogram, depending if we CHUNKIFY or not. */
		    if (!CHUNKIFY) {
			hist->update(new_label, true);
		    } else {
			std::vector<unsigned long> to_insert = chunkify((unsigned char *)new_label_str.c_str(), CHUNK_SIZE);
			for (std::vector<unsigned long>::iterator ti = to_insert.begin(); ti != to_insert.end(); ++ti)
			    hist->update(*ti, true);
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label of vertex (" << vertex.id() << "): " << new_label << std::endl;
#endif
		    /* Update the vertex's label*/
		    nl.lb[gcontext.iteration] = new_label;
		    nl.tm[gcontext.iteration] = neighborhood[0].tme[gcontext.iteration - 1];
		    vertex.set_data(nl);
		    /* Update its out-going edges.*/
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[gcontext.iteration] = new_label;
			/* Time stamp is the same as the vertex's timestamp,
			 * i.e., the smallest one among all in-coming neighbors. */
			el.tme[gcontext.iteration] = neighborhood[0].tme[gcontext.iteration - 1];
			out_edge->set_data(el);
		    }
		}
		/* Always schedule itself for the next iteration,
		 * until the base graph is completely processed. */
		if (gcontext.scheduler != NULL) {
		    if (gcontext.iteration < K_HOPS)
			/* Do not schedule for the next iteration during the K_HOPSth iteration
			 * because all nodes in the base graph should have been processed. */
			gcontext.scheduler->add_task(vertex.id());
		}
            /* Deal with streaming nodes/edges. */
	    } else {
		/* We first check if the node is a new node or not so that we can do some initialization.
		 * The node is new if any of its edges marks the node new. */
		bool is_new = false;
		for (int i = 0; i < vertex.num_outedges(); i++) {
		    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
		    EdgeDataType el = out_edge->get_data();
		    if (el.new_src)
			is_new = true;
		}
		if (!is_new) {
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			if (el.new_dst)
			    is_new = true;
		    }
		}
		/* Handle the case if the node is new. */
		/* Every new streamed node will only run the following code EXACTLY once.*/
		if (is_new) {
		    /* Initialize the new node. */
		    if (vertex.num_inedges() == 0) {
			/* If this new node is a new leaf node:
			 * - We popular all of its label and the label of its outedges.
			 * - We will not schedule this node for next iteration, unless new edges are associated with it later.
			 * - Mark the node as a leaf node. */
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Processing new leaf vertex: " << vertex.id() << std::endl;
#endif
			graphchi_edge<EdgeDataType> * out_edge = vertex.random_outedge(); /* The node must have at least one outedge. */
			assert(out_edge != NULL);
			EdgeDataType el = out_edge->get_data();
			VertexDataType nl;
			nl.lb[0] = el.src[0];
			nl.tm[0] = 0;
			/* Since the node has no incoming edges, all of its labels 
			 * are the same as the initial label. All of its timestamps
			 * are set to 0. */
			for (int i = 1; i < K_HOPS + 1; i++) {
			    nl.lb[i] = nl.lb[0];
			    nl.tm[i] = 0;
			}
			nl.is_leaf = true;
			/* Update node label. */
			vertex.set_data(nl);
			/* Populate the histogram for all its labels (hops). */
			for (int i = 0; i < K_HOPS + 1; i++) {
			    hist->decay(SFP);
			    hist->update(nl.lb[i], false);
			}
			/* Populate the labels to all of its out-going edges. */
			for (int i = 0; i < vertex.num_outedges(); i++) {
			    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			    EdgeDataType el = out_edge->get_data();
			    for (int j = 1; j < K_HOPS + 1; j++) {
				el.src[j] = nl.lb[j];
				/* Update the timestamps. */
				el.tme[j] = el.tme[j - 1];
			    }
			    el.new_src = false; /* Make sure every edge is marked as seen. */
			    out_edge->set_data(el);
			}
			return; /* We can immediately return for the vertex. */
		    } else {
			/* This new node is not a leaf node. Use the
			 * first inedge to get its original label. */
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Process new non-leaf vertex: " << vertex.id() << std::endl;
#endif
			graphchi_edge<EdgeDataType> * edge = vertex.inedge(0);
			VertexDataType nl = vertex.get_data();
			nl.lb[0] = edge->get_data().dst;
			nl.tm[0] = 0;
			vertex.set_data(nl);
			
			for (int i = 0; i < vertex.num_inedges(); i++) {
			    graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			    EdgeDataType el = in_edge->get_data();
			    /* For a new vertex, every in-edge should be a new edge with itr = 0. */
			    assert(el.itr == 0);
			    el.itr++; /* After this initialization, every new edge has "itr" value 1. */
			    el.new_dst = false; /* We make sure next iteration, we won't count the node as a new node. */
			    in_edge->set_data(el);
			}

			for (int i = 0; i < vertex.num_outedges(); i++) {
			    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			    EdgeDataType el = out_edge->get_data();
			    el.new_src = false; /* We make sure next iteration, we won't count the node as a new node. */
			    out_edge->set_data(el);
			}
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Vertex (" << vertex.id() << ") label: " << nl.lb[0] << std::endl;
#endif
			/* Populate histogram map. */
			hist->decay(SFP);
			hist->update(nl.lb[0], false);
		    }
		}
		/* The node is known to the system. */
		if (vertex.num_inedges() == 0) {
		    /* We first deal with leaf nodes that are scheduled.
		     * This leaf node has been initialized. Since non-leaf
		     * nodes cannot become leaf nodes, if an existing leaf
		     * node is scheduled, there must exist at least one
		     * out-edge of this node whose labels need to be populated. 
		     * We simply copy the leaf node info to all of its edges. */
		    VertexDataType nl = vertex.get_data();
		    assert(nl.is_leaf); /* Just a check to make sure the node is a leaf node. */
		    /* Note: some repetitive work may have occurred in the following loop.
		     * We have to do it because we don't know which edge has not been assigned. */
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			for (int j = 1; j < K_HOPS + 1; j++) {
			    el.src[j] = nl.lb[j];
			    el.tme[j] = el.tme[j - 1];
			}
			out_edge->set_data(el);
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "Streaming refreshes an existing leaf node: " << vertex.id() << std::endl;
#endif
		} else {
		    /* Handle nodes with incoming edges. */
		    VertexDataType nl = vertex.get_data();
		    if (nl.is_leaf)
			/* If this node used to be a leaf node. */
			nl.is_leaf = false;
		    /* In the case where a new edge occurs between two existing
		     * nodes, the edge needs to be sync'ed with the node.
		     * Some repetitive work may have occurred in the following
		     * loop; we have to do it because we don't know which edge
		     * has not been assigned yet. */
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			for (int j = 1; j < K_HOPS + 1; j++) {
			    el.src[j] = nl.lb[j];
			    el.tme[j] = nl.tm[j];
			}
			out_edge->set_data(el);
		    }
		    /* Change all incoming edges whose itr count is 0 to 1.
		     * At the same time, find the minimum itr among all inedges.*/
		    int min_itr = K_HOPS + 2; /* no itr value in our K_HOPS-hop case can be larger than K_HOPS + 2. */
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			if (el.itr == 0) {
			    el.itr++;
			    in_edge->set_data(el);
			}
			if (el.itr < min_itr)
			    min_itr = el.itr;
		    }
		    /* We check here since the minimum iteration value
		     * should be at least 1, but less than K_HOPS + 2. */
		    assert(min_itr > 0 && min_itr < K_HOPS + 2);
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "The min_itr of the vertex (" << vertex.id() << ") is: " << min_itr << std::endl;
#endif
		    if (min_itr == K_HOPS + 1)
			/* This node should not be scheduled again and do not run the rest of the logic.
			 * This node could be, for example, the source node of a new edge added. */
			return;
		    /* Now we update a new label. */
		    std::vector<EdgeDataType> neighborhood;
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			neighborhood.push_back(el);
			if (el.itr < K_HOPS + 1)
			    el.itr++; /* We increment edges whose itr value is less than K_HOPS + 1. */
			in_edge->set_data(el);
		    }
		    std::sort(neighborhood.begin(), neighborhood.end(), EdgeSorter(min_itr - 1));
		    /* First construct the string of the vertex itself. */
		    std::string new_label_str = "";
		    std::string first_str;
		    std::stringstream first_out;
		    first_out << vertex.get_data().lb[min_itr - 1];
		    first_str = first_out.str();
		    new_label_str += first_str; /* Use space to separate number strings. */
		    /* Then append neighborhood labels. */
		    for (std::vector<EdgeDataType>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
			if (min_itr == 1) { /* If the vertex is incorporating a new edge. */
			    std::string edge_str;
			    std::stringstream edge_out;
			    edge_out << it->edg;
			    edge_str = edge_out.str();
			    new_label_str += " " + edge_str;
			}
			std::string node_str;
			std::stringstream node_out;
			node_out << it->src[min_itr - 1];
			node_str = node_out.str();
			new_label_str += " " + node_str;
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label string of the vertex (" << vertex.id() << "): " << new_label_str << std::endl;
#endif
		    /* Relabel by hashing. */
		    unsigned long new_label = hash((unsigned char *)new_label_str.c_str());
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label of the vertex (" << vertex.id() << "): " << new_label << std::endl;
#endif
		    /* Populate the histogram. */
		    if (!CHUNKIFY) {
			hist->decay(SFP);
			hist->update(new_label, false);
		    } else {
			std::vector<unsigned long> to_insert = chunkify((unsigned char *)new_label_str.c_str(), CHUNK_SIZE);
			bool first = true;
			for (std::vector<unsigned long>::iterator ti = to_insert.begin(); ti != to_insert.end(); ++ti) {
			    if (first) {
				hist->decay(SFP);  /* Only decay once. */
				first = false;
			    }
			    hist->update(*ti, false);
			}
		    }
		    /* Update the vertex's label*/
		    nl.lb[min_itr] = new_label;
		    vertex.set_data(nl);
		    /* Update its out-going edges. We also decide
		     * if we want to schedule the vertex next. */
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[min_itr] = new_label;
			/* Time stamp is the smallest one among
			 * all of its in-coming neighbors. */
			el.tme[min_itr] = neighborhood[0].tme[min_itr - 1];
			/* Update the itr value. */
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Outgoing vertex (" << out_edge->vertex_id() << ") current itr:" << el.itr << std::endl;
#endif
			if (el.itr == K_HOPS + 1) {
			    /* We only need to update those nodes 
			     * that would not be scheduled otherwise. */
			    el.itr = min_itr + 1;
#ifdef DEBUG
			    logstream(LOG_DEBUG) << "Update outgoing vertex (" << out_edge->vertex_id() << ") itr to: " << el.itr << std::endl;
#endif
			}
			out_edge->set_data(el);
			
			if (min_itr < K_HOPS) {
			    /* Schedule the outgoing neighbor because
			     * it needs to update its label too. */
			    if (gcontext.scheduler != NULL)
				gcontext.scheduler->add_task(out_edge->vertex_id());			
			}
		    }
		    /* Now we decide if we want to schedule the node itself. */
		    if (min_itr < K_HOPS + 1) {
			/* Schedule itself because we haven't explore all hops yet. */
			if (gcontext.scheduler != NULL)
			    gcontext.scheduler->add_task(vertex.id());
		    }
		}
	    }
	}
	
	/* Called before an iteration starts. */
	void before_iteration(int iteration, graphchi_context &gcontext) {
	}

	/* Called after an iteration has finished. */
	void after_iteration(int iteration, graphchi_context &gcontext) {
#ifdef DEBUG
	    logstream(LOG_DEBUG) << "Current iteration: " << iteration << std::endl;
#endif
	    if (iteration == K_HOPS)
		std::base_graph_constructed = true;
	    if (std::no_new_tasks){
#ifdef DEBUG
		logstream(LOG_DEBUG) << "No new task at the moment...Let's see if we need to stop or wait..." << std::endl;
#endif
		if (std::stop) {
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "Everything is done!" << std::endl;
#endif
		    gcontext.set_last_iteration(iteration); /* Set this iteration as the last one. */
		    return;
		}
		pthread_barrier_wait(&std::stream_barrier);
		std::no_new_tasks = false;
#ifdef DEBUG
		logstream(LOG_DEBUG) << "No new tasks to run! But we have new streaming edges..." << std::endl;
#endif
		pthread_barrier_wait(&std::graph_barrier);
	    }
	}

	/* Called before an execution interval is started. */
	void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}

	/* Called after an execution interval has finished. */
	void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}

    };
}
