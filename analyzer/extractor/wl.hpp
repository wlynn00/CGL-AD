/*
 *
 * Author: Xueyuan Han <hanx@g.harvard.edu>
 *
 * Copyright (C) 2018 Harvard University
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

#include "graphchi_basic_includes.hpp"
#include "logger/logger.hpp"
#include "include/def.hpp"
#include "include/helper.hpp"

namespace graphchi {

	/**
	  * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
	  * class. The main logic is usually in the update function.
	  */
	struct WeisfeilerLehman : public GraphChiProgram<VertexDataType, EdgeDataType> {
		/* Get the singleton of histogram map. */
		Histogram* hist = Histogram::get_instance();

		/* Locks needed for updating histogram map. */
		std::mutex histogram_map_lock;

		/**
		 *  Vertex update function.
		 */
		void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
			/* Detected isolated vertex in the graph.
			 * The following code runs only in debugging mode to discover dirty data.
			 */
#ifdef DEBUG
			if (vertex.num_edges() <= 0) {
				logstream(LOG_DEBUG) << "Isolated vertex #"<< vertex.id() <<" detected." << std::endl;
				return;
			}
#endif
			/* Since we require synchronous updates, we define:
			 * 1. Even-numbered iterations: swap values between "curr" and "prev" on edges.
			 * 2. Odd-numbered iterations: update "curr" from "prev". 
			 */
			if (gcontext.iteration == 0) {
				/* On first iteration, initialize vertex label.
				 * This vertex label is its original label and should not be overwritten by later iterations.
				 */
				/* The "orig" of the node_label can be obtained from any outedge (from "prev") or inedge (from "curr") */
				struct node_label nl;

				graphchi_edge<EdgeDataType> * edge = vertex.random_outedge();
				if (edge == NULL) {
					/* If a vertex does not have any outedge, it can get its original label from any of its inedge, which it must have at least one inedge. */
					edge = vertex.inedge(0); /* Use the first inedge to get its original label. */
					nl.orig = edge->get_data().curr;
				} else
					nl.orig = edge->get_data().prev;
				nl.curr = nl.orig; /* For this iteration, the original and the current label are the same. */
				vertex.set_data(nl);

				/* Populate histogram map. */
				histogram_map_lock.lock();
				hist->insert_label(nl.orig);
				histogram_map_lock.unlock();
#ifdef DEBUG
				logstream(LOG_DEBUG) << "The original label of vertex #" << vertex.id() << " is: " << nl.orig << std::endl;
#endif
			} else if (gcontext.iteration % 2 == 1) { /* Odd-numbered iteration updates time and label values to "prev_time" and "prev". */
				/* During the 1st iteration, we include edge labels when relabeling. */
				std::vector<struct edge_label> neighborhood; /* We reuse edge_label struct to store neighbor vertex's label ("prev"), edge label ("edge"), and use "prev_time" for sorting. */
				/* We only consider immediate ancestors. */
				for (int i = 0; i < vertex.num_inedges(); i++) {
					graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
					
					struct edge_label neighbor;
					neighbor = in_edge->get_data();
					neighborhood.push_back(neighbor);
				}
				/* Labels are sorted based on the timestamps of the in_edges. */
				std::sort(neighborhood.begin(), neighborhood.end(), compareEdges);
				/* Appending neighborhood labels with the label of the current vertex itself.*/
				/* First construct the string of the vertex itself. */
				std::string new_label_str = "";
				std::string first_str;
				std::stringstream first_out;
				first_out << vertex.get_data().curr;
				first_str = first_out.str();
				new_label_str += first_str; /* Use space to separate number strings. */
				/* Then append neighborhood labels. */
				for (std::vector<struct edge_label>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
					if (gcontext.iteration == 1) { /* The first iteration includes edge labels. */
						std::string edge_str;
						std::stringstream edge_out;
						edge_out << it->edge;
						edge_str = edge_out.str();
						new_label_str += " " + edge_str;
					}
					std::string node_str;
					std::stringstream node_out;
					node_out << it->prev;
					node_str = node_out.str();
					new_label_str += " " + node_str;
				}
#ifdef DEBUG
				logstream(LOG_DEBUG) << "New label string of the vertex #" << vertex.id() << " is: " << new_label_str << std::endl;
#endif
				/* Relabel by hashing. */
				unsigned long new_label = hash((unsigned char *)new_label_str.c_str());
				/* Populate histogram map. */
				histogram_map_lock.lock();
				hist->insert_label(new_label);
				histogram_map_lock.unlock();
				/* Update the vertex's label*/
				struct node_label nl = vertex.get_data();
				nl.curr = new_label;
				vertex.set_data(nl);

				/* Update timestamp of the vertex to be the smallest among the neighbor timestamps for later iterations, if neighborhood exists.
				 * Since neighborhood is sorted. The first element is the smallest, if exists.
				 * Also update the label of the vertex to its outgoing edges to be used in the next iteration.
				 */
				for (int i = 0; i < vertex.num_outedges(); i++) {
					graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
					struct edge_label el = out_edge->get_data();
					if (neighborhood.size() > 0)
						el.curr_time = neighborhood[0].prev_time;
					el.curr = new_label;
					out_edge->set_data(el);
				}
#ifdef DEBUG
				if (neighborhood.size() > 0)
					logstream(LOG_DEBUG) << "Update time of the vertex #" << vertex.id() << " to: " << neighborhood[0].prev_time << std::endl;
				else
					logstream(LOG_DEBUG) << "Time of the vertex #" << vertex.id() << " unchanged." << std::endl;
#endif
			} else { /* Even-numbered iterations swap time and label values. */
				for (int i = 0; i < vertex.num_outedges(); i++) {
					graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
					struct edge_label el = out_edge->get_data();
					el.prev = el.curr;
					el.prev_time = el.curr_time;
					out_edge->set_data(el);
#ifdef DEBUG
					logstream(LOG_DEBUG) << "Swap out_edges of the vertex #" << vertex.id() << std::endl;
#endif
				}
			}
		}
		
		/**
		 * Called before an iteration starts.
		 */
		void before_iteration(int iteration, graphchi_context &gcontext) {
		}
	    
		/**
		 * Called after an iteration has finished.
		 */
		void after_iteration(int iteration, graphchi_context &gcontext) {
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Current Iteration: " << iteration << std::endl;
			hist->print_histogram();
#endif
		}
		
		/**
		 * Called before an execution interval is started.
		 */
		void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
		}
		
		/**
		 * Called after an execution interval has finished.
		 */
		void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
		}
		
	};

}