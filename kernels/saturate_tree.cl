#include "saturate_tree.hcl"
#include "common.hcl"

// Based on the following code:
// https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/summarizetree.cl

__kernel void saturate_tree(
    __global float *positions,
    __global volatile atomic_float *mass,
    __global volatile int *children,
    const int num_of_bodies,
    const int num_of_nodes,
    __global volatile int* bottom,
    __global atomic_int* body_count
)
{
#ifdef DEBUG
    if(get_global_id(0) == 0)
        DEBUG_PRINT(("----STATURATE TREE----"));
#endif
    __local volatile int localChild[ WORKGROUP_SIZE * NUMBER_OF_CELLS];
    int step_size = get_local_size(0) * get_num_groups(0);
    int l_id = get_local_id(0);
    int g_id = get_global_id(0);

    const int cbottom = *bottom;
    int missing = 0;

    int node_body_count;
    float mass_idx;
    float3 node_center;
    float node_mass;

    int node_idx = (cbottom & -WARP_SIZE) + get_global_id(0);
    if (node_idx < cbottom){
        node_idx += step_size;
    }
//    DEBUG_PRINT(("[%d:%d] %d to %d and %d",g_id,l_id,node_idx,num_of_bodies,num_of_nodes));

    while (node_idx <= (num_of_bodies + num_of_nodes))
    {
        if (missing == 0){
//            DEBUG_PRINT(("\t\t[%d] Init Node: %d\n",g_id,node_idx));

            node_center = (float3){0.0f,0.0f,0.0f};
            node_mass = 0.0f;
            node_body_count = 0;

            int used_child_idx = 0;

#pragma unroll NUMBER_OF_CELLS
            for (int c_idx = 0; c_idx < NUMBER_OF_CELLS; ++c_idx){
                int child = children[node_idx * NUMBER_OF_CELLS + c_idx];
//                DEBUG_PRINT(("\t\t\t Child Index: %d\n\t\t\tChild: %d\n",c_idx,child));



                if (child >= 0) { // Child points to existing node or body

                    if (c_idx != used_child_idx){ // Non null children get moved to the front
                        children[NUMBER_OF_CELLS * node_idx + c_idx] = -1;
                        children[NUMBER_OF_CELLS * node_idx + used_child_idx] = child;
                    }

                    localChild[ WORKGROUP_SIZE * missing + l_id] = child; // Cache children,

                    mass_idx = atomic_load_explicit(&mass[child],memory_order_seq_cst,memory_scope_device);

//                    DEBUG_PRINT(("\t\t\t\t[%d] Mass of Child: %f\n",g_id,mass_idx));

                    ++missing;

                    if (mass_idx >= 0.0f) { // Mass is available
//                        DEBUG_PRINT(("\t\t\t\t[%d] Child ready\n",g_id));

                        --missing;
                        if (child >= num_of_bodies){ // Child is a node
//                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count BEFORE: %d\n",g_id,node_body_count));

                            node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device) - 1;

//                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count AFTER: %d\n",g_id,node_body_count));
                        }
                        node_mass += mass_idx;
                        node_center.x += positions[child * 3 + 0] * mass_idx;
                        node_center.y += positions[child * 3 + 1] * mass_idx;
                        node_center.z += positions[child * 3 + 2] * mass_idx;
                    }
                    ++used_child_idx;
                }
            }
            mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            node_body_count += used_child_idx; // += "Number of existing nodes/bodies"
        }

        if (missing != 0){ // Mass wasn't available for at least one child
//            DEBUG_PRINT(("\t\t[%d:%d] Missing %d - Not zero!",g_id,l_id,missing));
            do {
                int child = localChild[(missing - 1) * WORKGROUP_SIZE + l_id]; // Because of the previous code block, this will yield the missing child
                // Missing > 0, therefore the child that gets cached here is guaranteed to be missing!

                mass_idx = atomic_load_explicit(&mass[child],memory_order_seq_cst,memory_scope_device);

                if (mass_idx >= 0.0f){ // Node
                    --missing;

                    if (child >= num_of_bodies){
                        node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device) - 1;
                    }

                     node_mass += mass_idx;
                     node_center.x += positions[child * 3 + 0] * mass_idx;
                     node_center.y += positions[child * 3 + 1] * mass_idx;
                     node_center.z += positions[child * 3 + 2] * mass_idx;

                }

            } while((mass_idx >= 0.0f) && (missing != 0));
            // This will loop if one child becomes available and we are still missing children

            // If child is not available, or we aren't missing any more children, due to thread divergence these threads will slow their progress

//            DEBUG_PRINT(("\t\t [%d:%d] Mass: %f\t Node Mass: %f\n",g_id,l_id,mass_idx,node_mass));
        }

        if (missing == 0){ // If no failure happened, continue
//            DEBUG_PRINT(("[%d]\tMissing is zero\n\tNode Index: %d\n\tNode Mass: %f\n\tNode center of mass: (%f,%f,%f)\n",g_id,node_idx,node_mass,node_center.x,node_center.y,node_center.z));


            atomic_store_explicit(&body_count[node_idx],node_body_count,memory_order_seq_cst,memory_scope_device); // Necessary for later sorting only!


            mass_idx = 1.0f / node_mass;

            positions[node_idx * 3 + 0] = node_center.x * mass_idx;
            positions[node_idx * 3 + 1] = node_center.y * mass_idx;
            positions[node_idx * 3 + 2] = node_center.z * mass_idx;

            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst,memory_scope_device);
            atomic_store_explicit(&(mass[node_idx]),node_mass,memory_order_seq_cst,memory_scope_device);

//            DEBUG_PRINT(("\t\tNext node: %d\n",node_idx));

            node_idx += step_size; // Continue with next cell
            // If a thread fails with a node, it continues trying to get all missing childrne in the next loop
        }
//        DEBUG_PRINT(("\t\tNext node: %d\n\tMissing: %d",node_idx,missing));

    }
}

__kernel void saturate_tree_ext(
    __global float *positions,
    __global volatile atomic_float *mass,
    __global volatile int *children,
    const int num_of_bodies,
    const int num_of_nodes,
    __global volatile int* bottom,
    __global atomic_int* body_count
)
{
#ifdef DEBUG
    if(get_global_id(0) == 0)
        DEBUG_PRINT(("----STATURATE TREE----"));
#endif
    __local volatile int localChild[ WORKGROUP_SIZE * NUMBER_OF_CELLS];
    int step_size = get_local_size(0) * get_num_groups(0);
    int l_id = get_local_id(0);
    int g_id = get_global_id(0);

    const int cbottom = *bottom;
    int missing = 0;

    int node_body_count;
    float mass_idx;
    float3 node_center;
    float node_mass;

    int node_idx = (cbottom & -WARP_SIZE) + get_global_id(0);
    if (node_idx < cbottom){
        node_idx += step_size;
    }
//    DEBUG_PRINT(("[%d:%d] %d to %d and %d",g_id,l_id,node_idx,num_of_bodies,num_of_nodes));

    while (node_idx <= (num_of_bodies + num_of_nodes))
    {
        if (missing == 0){
//            DEBUG_PRINT(("\t\t[%d] Init Node: %d\n",g_id,node_idx));

            node_center = (float3){0.0f,0.0f,0.0f};
            node_mass = 0.0f;
            node_body_count = 0;

            int used_child_idx = 0;

#pragma unroll NUMBER_OF_CELLS
            for (int c_idx = 0; c_idx < NUMBER_OF_CELLS; ++c_idx){
                int child = children[node_idx * NUMBER_OF_CELLS + c_idx];
//                DEBUG_PRINT(("\t\t\t Child Index: %d\n\t\t\tChild: %d\n",c_idx,child));


                 if (child >= 0) { // Child points to existing node or body

                    if (c_idx != used_child_idx){ // Non null children get moved to the front
                        children[NUMBER_OF_CELLS * node_idx + c_idx] = -1;
                        children[NUMBER_OF_CELLS * node_idx + used_child_idx] = child;
                    }

                    localChild[ WORKGROUP_SIZE * missing + l_id] = child; // Cache children,

                    mass_idx = atomic_load_explicit(&mass[child],memory_order_seq_cst,memory_scope_device);

//                    DEBUG_PRINT(("\t\t\t\t[%d] Mass of Child: %f\n",g_id,mass_idx));

                    ++missing;

                    if (mass_idx >= 0.0f) { // Mass is available
//                        DEBUG_PRINT(("\t\t\t\t[%d] Child ready\n",g_id));

                        --missing;
                        if (child >= num_of_bodies){ // Child is a node
//                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count BEFORE: %d\n",g_id,node_body_count));

                            node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device) - 1;

//                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count AFTER: %d\n",g_id,node_body_count));
                        }


                        node_mass += mass_idx;
                        node_center.x += positions[child * 3 + 0] * mass_idx;
                        node_center.y += positions[child * 3 + 1] * mass_idx;
                        node_center.z += positions[child * 3 + 2] * mass_idx;

                        int max_loop = child > num_of_bodies ? 0 : LEAF_CAP;

                        for (int i = 0; i < max_loop && children[child * NUMBER_OF_CELLS + i] >= 0; ++i){
                            int c_idx = children[child * NUMBER_OF_CELLS + i];
                            mass_idx = atomic_load_explicit(&mass[c_idx],memory_order_seq_cst,memory_scope_device);
                            node_mass += mass_idx;
                            node_center.x += positions[c_idx * 3 + 0] * mass_idx;
                            node_center.y += positions[c_idx * 3 + 1] * mass_idx;
                            node_center.z += positions[c_idx * 3 + 2] * mass_idx;
                            ++node_body_count;
                        }
                    }
                    ++used_child_idx;
                }
            }
            mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            node_body_count += used_child_idx; // += "Number of existing nodes/bodies"
        }

        if (missing != 0){ // Mass wasn't available for at least one child
//            DEBUG_PRINT(("\t\t[%d:%d] Missing %d - Not zero!",g_id,l_id,missing));
            do {
                int child = localChild[(missing - 1) * WORKGROUP_SIZE + l_id]; // Because of the previous code block, this will yield the missing child
                // Missing > 0, therefore the child that gets cached here is guaranteed to be missing!

                mass_idx = atomic_load_explicit(&mass[child],memory_order_seq_cst,memory_scope_device);

                if (mass_idx >= 0.0f){ // Node
                    --missing;

                    if (child >= num_of_bodies){
                        node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device) - 1;
                    }

                    node_mass += mass_idx;
                    node_center.x += positions[child * 3 + 0] * mass_idx;
                    node_center.y += positions[child * 3 + 1] * mass_idx;
                    node_center.z += positions[child * 3 + 2] * mass_idx;
                    // Because a child is missing and node leafs can NEVER be missing, we don't have to bother with body clusters
                }

            } while((mass_idx >= 0.0f) && (missing != 0));
            // This will loop if one child becomes available and we are still missing children

            // If child is not available, or we aren't missing any more children, due to thread divergence these threads will slow their progress

//            DEBUG_PRINT(("\t\t [%d:%d] Mass: %f\t Node Mass: %f\n",g_id,l_id,mass_idx,node_mass));
        }

        if (missing == 0){ // If no failure happened, continue
//            DEBUG_PRINT(("[%d]\tMissing is zero\n\tNode Index: %d\n\tNode Mass: %f\n\tNode center of mass: (%f,%f,%f)\n",g_id,node_idx,node_mass,node_center.x,node_center.y,node_center.z));


            atomic_store_explicit(&body_count[node_idx],node_body_count,memory_order_seq_cst,memory_scope_device); // Necessary for later sorting only!


            mass_idx = 1.0f / node_mass;

            positions[node_idx * 3 + 0] = node_center.x * mass_idx;
            positions[node_idx * 3 + 1] = node_center.y * mass_idx;
            positions[node_idx * 3 + 2] = node_center.z * mass_idx;

            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst,memory_scope_device);
            atomic_store_explicit(&(mass[node_idx]),node_mass,memory_order_seq_cst,memory_scope_device);

//            DEBUG_PRINT(("\t\tNext node: %d\n",node_idx));

            node_idx += step_size; // Continue with next cell
            // If a thread fails with a node, it continues trying to get all missing childrne in the next loop
        }
//        DEBUG_PRINT(("\t\tNext node: %d\n\tMissing: %d",node_idx,missing));

    }
}