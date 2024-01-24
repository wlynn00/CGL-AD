#ifndef __EXTERN_HPP__
#define __EXTERN_HPP__

#include <pthread.h>

namespace std {
    extern pthread_barrier_t graph_barrier;
    extern pthread_barrier_t stream_barrier;
    extern int stop;
    extern bool base_graph_constructed;
    extern bool no_new_tasks;
}

#endif /* __EXTERN_HPP__ */
