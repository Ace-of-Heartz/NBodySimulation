#include "saturate_tree.hcl"
#include "common.hcl"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable




//TODO Refactor bodies_n_nodes so that it contains the position AND MASS -> Revisit initialization of particle!
//TODO Implement body_count!
//TODO Test this!
__kernel void saturate_tree(
    __global float *bodies_n_nodes,
    __global atomic_float *mass,
    __global int *children, //Locks for each node in the tree
    const int num_of_bodies,
    const int num_of_nodes,
    __global volatile int* bottom,
    __global atomic_int* body_count
)
{
    __local volatile int localChild[WORKGROUP_SIZE * NUMBER_OF_CELLS];
    int step_size = get_global_size(0); //get_local_size(0) * get_num_groups(0); too
    int l_id = get_local_id(0);
    int g_id = get_global_id(0);


    const int cbottom = *bottom;
    int missing = 0;



    int node_body_count = 0;
    float mass;
    float4 node_center_n_mass;

// TODO Recheck bottom definition!
    int node_idx = (cbottom & -WARP_SIZE) + get_global_id(0);
    if (node_idx < cbottom){
        node_idx += step_size;
    }

    while (node_idx <= num_of_nodes)
    {
        if (missing == 0){
            DEBUG_PRINT(("\t\t[%d] Init Node\n",g_id));
            node_center_n_mass = (float4){0.0f,0.0f,0.0f,0.0f};
            int used_child_idx = 0;

            for (int c_idx = 0; c_idx < NUMBER_OF_CELLS; ++c_idx){
                int child = children[node_idx * NUMBER_OF_CELLS + c_idx];
                DEBUG_PRINT(("\t\t\t Child Index: %d\n\t\t\tChild: %d\n",c_idx,child));



                if (child >=0) {
                    // "Used" == Child points to existing node or body

                    if (child != used_child_idx){
                        children[NUMBER_OF_CELLS * node_idx + c_idx] = -1;
                        children[NUMBER_OF_CELLS * node_idx + used_child_idx] = child;
                    }

                    localChild[WORKGROUP_SIZE * missing + get_local_id(0)] = child;

                    mass = atomic_load_explicit(&bodies_n_nodes[child * BODY_DATA_LEN + 3],memory_order_seq_cst,memory_scope_device);

                    DEBUG_PRINT(("\t\t\t\t[%d] Mass of Child: %f\n",g_id,mass));

                    ++missing;

                    if (mass >= 0.0f) {
                        DEBUG_PRINT(("\t\t\t\t[%d] Child ready\n",g_id));

                        --missing;
                        if (child >= num_of_bodies){
                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count BEFORE: %d\n",g_id,node_body_count));

                            node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device) - 1;

                            DEBUG_PRINT(("\t\t\t\t[%d] Node Body Count AFTER: %d\n",g_id,node_body_count));
                        }
                        node_center_n_mass.w += mass;
                        node_center_n_mass.x += bodies_n_nodes[child * BODY_DATA_LEN + 0] * mass;
                        node_center_n_mass.y += bodies_n_nodes[child * BODY_DATA_LEN + 1] * mass;
                        node_center_n_mass.z += bodies_n_nodes[child * BODY_DATA_LEN + 2] * mass;
                    }
                    ++used_child_idx;
                }
            }
            node_body_count += used_child_idx
        }

        if (missing != 0){
            DEBUG_PRINT(("\t\t[%d] Missing %d - Not zero!",missing));
            do {
                int child = localChild[(missing - 1) * WORKGROUP_SIZE + l_id];

                mass = atomic_load_explicit(&(bodies_n_nodes[child * BODY_DATA_LEN + 3]),memory_order_seq_cst,memory_scope_device);

                if (mass >= 0.0f){
                    --missing;

                    if (child >= num_of_bodies){
                        node_body_count += atomic_load_explicit(&body_count[child],memory_order_seq_cst,memory_scope_device);
                    }

                     node_center_n_mass.w += mass;
                     node_center_n_mass.x += bodies_n_nodes[child * BODY_DATA_LEN + 0] * mass;
                     node_center_n_mass.y += bodies_n_nodes[child * BODY_DATA_LEN + 1] * mass;
                     node_center_n_mass.z += bodies_n_nodes[child * BODY_DATA_LEN + 2] * mass;
                       
                }

            } while((mass >= 0.0f) && (missing != 0));

            DEBUG_PRINT(("\t\t [%d:%d] Mass: %f",g_id,l_id,mass));
        }

    }
    
    if (missing == 0){
        DEBUG_PRINT(("\t Missing is zero\n"));
        
        atomic_store_explicit(&body_count[node_idx],node_body_count,memory_order_seq_cst,memory_scope_device);
        mass = 1.0f / node_center_n_mass.w;
        
        bodies_n_nodes[node_idx * BODY_DATA_LEN + 0] = node_center_n_mass.x * mass;
        bodies_n_nodes[node_idx * BODY_DATA_LEN + 1] = node_center_n_mass.y * mass;
        bodies_n_nodes[node_idx * BODY_DATA_LEN + 2] = node_center_n_mass.z * mass;
        
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst,memory_scope_device);
        atomic_store_explicit(&(bodies_n_nodes[node_idx * BODY_DATA_LEN + 3]),node_center_n_mass.w,memory_order_seq_cst,memory_scope_device);
        
        node_idx += step_size; // Continue with next cell
        DEBUG_PRINT(("\t\tNext node: %d\n",node_idx));

    }

}