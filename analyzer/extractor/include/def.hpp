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
#ifndef def_hpp
#define def_hpp

struct edge_label {
	unsigned long prev;
	unsigned long curr;
	unsigned long edge;
	int orig_time;
	int prev_time;
	int curr_time;
};

struct node_label {
	unsigned long orig;
	unsigned long curr;
};

typedef node_label VertexDataType; /* VertexDataType stores the original label of the vertex and a current label for each iteration. */

/* edge_label sttuct serves multiple purposes.
 * 1. "prev":
 * 		- Initalizes source vertex label at the beginning.
 * 		- Stores the label of the previous iteration of source vertex (so destination vertex can use the label to compute its new label) 
 * 2. "curr":
 * 		- Initalizes destination vertex label at the beginning.
 * 		- Stores the label for the current iteration of source vertex (so we can swap "prev" and "curr" in the next iteration)
 * 3. "edge" stores the original edge label of an edge. This value does not changes with iterations.
 * 4. "orig_time" stores the original timestamp of the edge.
 * 5. "prev_time" and "curr_time" are similar to "prev" and "curr" but concern timestamps.
 * 		- At very beginning, "prev_time" and "curr_time" are set to be the same as "orig_time".
 * "prev" and "curr" are needed because we simulate synchronous execution on asynchronous GraphChi graph processing system.
 */
typedef edge_label EdgeDataType;

struct {
	bool operator()(struct edge_label a, struct edge_label b) {
		return a.prev_time < b.prev_time;
	}
} compareEdges;

#endif