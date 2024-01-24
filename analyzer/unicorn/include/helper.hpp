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
#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>
/* Header file from GraphChi. */
#include "logger/logger.hpp"
/* Header file from Unicorn. */
#include "def.hpp"

namespace graphchi {
    /* Deterministically hash character strings to
     * a unique unsigned long integer. */
    unsigned long hash(unsigned char *str) {
        unsigned long hash = 5381;
        int c;

        while (c = *str++)
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        return hash;
    }

    /* Customized parse function to parse edge labels
     * from edgelist-formatted file (for base graph). */
    void parse(EdgeDataType &e, const char *s) {
        char *ss = (char *) s;
        char delims[] = ":";
        unsigned char *t;
        char *k;

        e.itr = 0; /* At the beginning, itr count is always 0. */
        /* GraphChi WL handle base grpah different, so @new_src
	 * and @new_dst fields are not used in base graph. */
        e.new_src = false;
        e.new_dst = false;
	
        t = (unsigned char *)strtok(ss, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Source label is missing." << std::endl;
        assert(t != NULL);
        e.src[0] = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Destination label does is missing." << std::endl;
        assert (t != NULL);
        e.dst = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Edge label is missing." << std::endl;
        assert (t != NULL);
        e.edg = hash(t);

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Timestamp is missing." << std::endl;
        assert (k != NULL);
        e.tme[0] = std::strtoul(k, NULL, 10);
#ifdef DEBUG
        k = strtok(NULL, delims);
        if (k != NULL)
            logstream(LOG_ERROR) << "Extra info is ignored." << std::endl;
#endif

        return;
    }

    /* Chunk a string into a vector of hashed 
     * substrings (hashed to unsigned long)
     * to be inserted into the histogram. */
    std::vector<unsigned long> chunkify(unsigned char *s, int chunk_size) {
        char *ss = (char *) s;
        char delims[] = " ";
        char *t;
        int counter = 0;
        std::string to_hash = "";
        std::vector<unsigned long> rtn;
        bool first = true;

        /* chunk_size must be larger than 1. */
        assert(chunk_size > 1);

        t = strtok(ss, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "The string to be chunked must be non-empty." << std::endl;
        assert(t != NULL);

        /* We explain the following logic using an example.
         * If we have a string: 12334 456 76634 4546 2345, and chunk_size is set to 2.
         * We produce the following substrings:
         * 1. 12334 45
         * 2.  76634 4546
         * 3.  2345
         * Note that there is a space in the front of substring 2 and substring 3.
         * @chunk_size is the maximum size of the chunk. Note that substring 3 has only 1 string.
         * We hash each substring and push it into the vector to return. */
        while (t != NULL) {
            std::string str(t);
            if (first) {
                to_hash += str;
                first = false;
            } else
                to_hash += " " + str;
            counter++;
            if (counter == chunk_size) {
                rtn.push_back(hash((unsigned char *)to_hash.c_str()));
                counter = 0;
                to_hash = "";
            }
            t = strtok(NULL, delims);
        }
        /* Handle the leftover (last substring) that might be < chunk_size. */
        if (to_hash.length() > 0) {
            rtn.push_back(hash((unsigned char *)to_hash.c_str()));
        }
        return rtn;
    }

    /* Compare two edges based on one of the timestamp specified by @pos.
     * This function is used to sort the edges. */
    bool compareEdges(EdgeDataType a, EdgeDataType b, int pos) {
        return a.tme[pos] < b.tme[pos];
    }
    
    /* This class is used to sort EdgeDataType based on its "tme" value at position n. */
    class EdgeSorter{
        int pos;

        public:
            EdgeSorter(int pos) {
                this->pos = pos;
            }

            bool operator()(EdgeDataType a, EdgeDataType b) const {
                return compareEdges(a, b, pos);
            }
    };

}

#endif /* __HELPER_HPP__ */
