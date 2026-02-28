#pragma once
#include <cstdio>

#define ANDROID_LOG_DEBUG 3

#define __android_log_print(prio, tag, ...) \
    do { fprintf(stderr, "[%s] ", tag); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
