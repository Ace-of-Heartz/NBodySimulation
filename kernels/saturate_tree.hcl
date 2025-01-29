__kernel void saturate_tree(
    __global float *positions,
    __global volatile atomic_float *mass,
    __global volatile int *children,
    const int num_of_bodies,
    const int num_of_nodes,
    __global volatile int* bottom,
    __global atomic_int* body_count
);