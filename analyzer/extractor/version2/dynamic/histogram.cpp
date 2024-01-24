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
#include "include/histogram.hpp"

Histogram* Histogram::histogram;

Histogram* Histogram::get_instance() {
	if (!histogram) histogram = new Histogram();
	return histogram;
}

Histogram::~Histogram(){
	this->size = 0;
	delete histogram;
}

/*!
 * @brief Insert @label to the histogram_map if it does not exist; otherwise, update the mapped value.
 *
 */
void Histogram::insert_label(unsigned long label) {
	std::pair<std::map<unsigned long, int>::iterator, bool> rst;
	rst = this->histogram_map.insert(std::pair<unsigned long, int>(label, 1));
	if (rst.second == false) {
#ifdef DEBUG
		logstream(LOG_DEBUG) << "The label " << label << " is already in the map. Updating the value only." << std::endl;
#endif
		rst.first->second++;
	} else {
		this->size++;
	}

	return;
}

/*!
 * @brief Print the histogram map for debugging.
 *
 */
void Histogram::print_histogram() {
	std::map<unsigned long, int>::iterator it;
	logstream(LOG_DEBUG) << "Printing histogram map to the console..." << std::endl;
	for (it = this->histogram_map.begin(); it != this->histogram_map.end(); it++)
		logstream(LOG_DEBUG) << "[" << it->first << "]->" << it->second << "  ";
	return;
}