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
#ifndef helper_hpp
#define helper_hpp

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>

#include "logger/logger.hpp"
#include "def.hpp"

namespace graphchi {
	
	/*!
	 * @brief Deterministically hash character strings to a unique unsigned long integer.
	 *
	 */
	unsigned long hash(unsigned char *str) {
		unsigned long hash = 5381;
		int c;

		while (c = *str++)
			hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
		return hash;
	}

	/*!
	 * @brief Customized parse funtion to parse edge values from edgelist-formatted file.
	 *
	 */
	void parse(edge_label &x, const char *s) {
	    char *ss = (char *) s;
	    char delims[] = ":";
	    unsigned char *t;
	    char *k;
	    
	    t = (unsigned char *)strtok(ss, delims);
	    if (t == NULL)
	        logstream(LOG_FATAL) << "Source label does not exist." << std::endl;
	    assert(t != NULL);
	    x.prev = hash(t);
	    
	    t = (unsigned char *)strtok(NULL, delims);
	    if (t == NULL)
	        logstream(LOG_FATAL) << "Destination label does not exist." << std::endl;
	    assert (t != NULL);
	    x.curr = hash(t);

	    t = (unsigned char *)strtok(NULL, delims);
	    if (t == NULL)
	        logstream(LOG_FATAL) << "Edge label does not exist." << std::endl;
	    assert (t != NULL);
	    x.edge = hash(t);

	    k = strtok(NULL, delims);
	    if (k == NULL)
	        logstream(LOG_FATAL) << "Time label does not exist." << std::endl;
	    assert (k != NULL);
	    x.orig_time = std::stoi(k);
	    x.prev_time = x.orig_time;
	    x.curr_time = x.orig_time;

	    k = strtok(NULL, delims);
	    if (k != NULL)
	        logstream(LOG_FATAL) << "Extra info will be ignored." << std::endl;

	    return;
	}

}

#endif