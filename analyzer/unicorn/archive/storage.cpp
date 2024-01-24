#include "include/def.hpp"
#include "include/storage.hpp"
#include "include/kissdb.h"
#include "logger/logger.hpp"

int kiss_storage::init_storage(const char *path) {
	return KISSDB_open(&this->db, path, KISSDB_OPEN_MODE_RWCREAT, 1000000, sizeof(unsigned long), sizeof(struct hist_elem));
}

int kiss_storage::get_item(const unsigned long *key, struct hist_elem *value) {
	return KISSDB_get(&this->db, key, value);
}

int kiss_storage::put_item(const unsigned long *key, struct hist_elem *value) {
	return KISSDB_put(&this->db, key, value);
}

void kiss_storage::close_storage() {
	KISSDB_close(&this->db);
}
