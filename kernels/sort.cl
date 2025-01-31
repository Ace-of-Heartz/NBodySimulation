
// Based on the following code:
// https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/sort.cl
// Not many things can be changed up due to the simplicity of the idea behind it after understanding
__kernel void sort(
    __global int *children,
    const int num_of_bodies,
    const int num_of_nodes,
    __global int *body_count,
    __global volatile atomic_int* start,
    __global int *bottom,
    __global int *sorted // Sorted indexes pointing to bodies
)
{
    int step_size = get_global_size(0);

    int cell = num_of_bodies + num_of_nodes + 1 - step_size + get_global_id(0);

    while (cell >= *bottom){
        int cell_start = atomic_load_explicit(&start[cell],memory_order_acquire,memory_scope_device);

        if (cell_start >= 0) { // Already know the index of the first element in the subtree
            for (int c_path = 0; c_path < NUMBER_OF_CELLS; ++c_path ){ // Iterate over children
                int child = children[c_path + cell * NUMBER_OF_CELLS];

                if (child >= num_of_bodies){ // Node
                    atomic_store_explicit(&start[child],cell_start,memory_order_release,memory_scope_device);

                    // Fence to flag that atomic variable has been update
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst,memory_scope_device);

                    cell_start += body_count[child];
                }
                else if (child >= 0){ // Body
                    sorted[cell_start] = child;

                    ++cell_start;
                }
            }

            cell -= step_size;
        }
    }
}

__kernel void sort_ext(
    __global int *children,
    const int num_of_bodies,
    const int num_of_nodes,
    __global int *body_count,
    __global volatile atomic_int* start,
    __global int *bottom,
    __global int *sorted // Sorted indexes pointing to bodies
)
{
    int step_size = get_global_size(0);

    int cell = num_of_bodies + num_of_nodes + 1 - step_size + get_global_id(0);

    while (cell >= *bottom){
        int cell_start = atomic_load_explicit(&start[cell],memory_order_acquire,memory_scope_device);

        if (cell_start >= 0) { // Already know the index of the first element in the subtree
            for (int c_path = 0; c_path < NUMBER_OF_CELLS; ++c_path ){ // Iterate over children
                int child = children[c_path + cell * NUMBER_OF_CELLS];

                if (child >= num_of_bodies){ // Node
                    atomic_store_explicit(&start[child],cell_start,memory_order_release,memory_scope_device);

                    // Fence to flag that atomic variable has been update
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst,memory_scope_device);

                    cell_start += body_count[child];
                }
                else if (child >= 0){ // Body
                    sorted[cell_start] = child;

                    ++cell_start;

                    for (int i = 0; i < LEAF_CAP && children[child * NUMBER_OF_CELLS + i] >= 0; ++i){

                        sorted[cell_start] = children[child * NUMBER_OF_CELLS + i];
                        ++cell_start;
                    }
                }
            }

            cell -= step_size;
        }
    }
}
