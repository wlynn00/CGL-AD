/*
 *
 * Author: Xueyuan Han <hanx@g.harvard.edu>
 *
 * Copyright (C) 2018 Harvard University, University of Cambridge
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2, as
 * published by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 */
#ifndef storage_hpp
#define storage_hpp

#include "def.hpp"
#include "kissdb.h"

/* Abstract Storage Class. */
class Storage {
	public:
		/* Open a storage file. */
		virtual int init_storage(const char *path) = 0;
		virtual int get_item(const unsigned long *key, struct hist_elem *value) = 0;
		virtual int put_item(const unsigned long *key, struct hist_elem *value) = 0;
		virtual void close_storage() = 0;
};

/* Concrete classes. */
class kiss_storage: public Storage {
	public:
		int init_storage(const char *path);
		int get_item(const unsigned long *key, struct hist_elem *value);
		int put_item(const unsigned long *key, struct hist_elem *value);
		void close_storage();

	private:
		KISSDB db;
};

extern kiss_storage ks;

#include "storage.cpp"

#endif
