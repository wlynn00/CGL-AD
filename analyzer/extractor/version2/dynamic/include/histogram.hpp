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
#ifndef histogram_hpp
#define histogram_hpp

#include <iostream>
#include <map>
#include <vector>
#include "logger/logger.hpp"

/* We use singleton design to create a single instance of a histogram.
 * This is not thread-safe. A proper locking mechanism is needed.
 * 
 * Current implementation uses a simple ordered Map.
 */

class Histogram {
public:
	static Histogram* get_instance();
	~Histogram();
	void insert_label(unsigned long label);
	void print_histogram();

private:
	static Histogram* histogram;

	Histogram(int size = 0) {
		this->size = size;
	}

	std::map<unsigned long, int> histogram_map;

	int size; /* Number of elements in the histogram_map. */

};
#include "histogram.cpp"

#endif