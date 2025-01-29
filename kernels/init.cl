#include "common.hcl"
__kernel void init(
    __global float* positions,
    __global float* mass,
    __global int* children,
    __global float* error,
    __global float* boundaries,
    __global int* bottom,
    const int num_of_bodies,
    const int num_of_nodes,
    const unsigned max_children,
    __global int* body_depth
)
{
    int g_id = get_global_id(0);

    for (int i = g_id; i < max_children; i += get_global_size(0)){
        children[i] = DEFAULT; // Set all children to default value
    }

    for (int i = g_id; i < num_of_bodies; i += get_global_size(0)){
        error[i] = 0;
        body_depth[i] = 0;
    }

    if(get_global_id(0) == 0)
    {
        positions[(num_of_bodies + num_of_nodes) * 3 + 0] = (boundaries[0] + boundaries[3]) / 2.0f;
        positions[(num_of_bodies + num_of_nodes) * 3 + 1] = (boundaries[1] + boundaries[4]) / 2.0f;
        positions[(num_of_bodies + num_of_nodes) * 3 + 2] = (boundaries[2] + boundaries[5]) / 2.0f;
        mass[num_of_nodes + num_of_bodies] = -1.0f;
        *bottom = num_of_nodes + num_of_bodies;
    }


}