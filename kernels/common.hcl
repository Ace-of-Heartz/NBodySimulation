#ifndef COMMON_HCL
#define COMMON_HCL

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


#define LOCKED -2
#define DEFAULT -1

#define LEAF_CAP 8
#define NUMBER_OF_CELLS 8
#define WARP_SIZE 16 // Not sure about this

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

typedef enum NumericalMethod{
    Leapfrog = 1,
    Euler = 2
} NumericalMethod;


#endif
