#pragma once
#include <assert.h>
#include <math.h>

void assert_close(volatile float a, volatile float b) {
    if (fabsf(a - b) > 1e-4) {
        fprintf(stderr, "assert_close: expected %f but got %f\n", b, a);
        assert(false);
    }
}