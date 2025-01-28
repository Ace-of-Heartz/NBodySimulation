#include "calculate_forces.hcl"
#include "common.hcl"

//TODO: If this doesn't work properly, make it only global memory / remove workspace related stuff

__kernel void calculate_force_global(
    __global float* positions,
    __global float* mass,
    __global float* vels,
    __global float* accs,
    __global int* children,
    const float theta,
    const float eps,
    const int num_of_bodies,
    const int num_of_nodes,
    const float G,
    const unsigned dt
)
{
    int g_id = get_global_id(0);

    int p_idx = g_id;

    while (p_idx < num_of_bodies){
        float3 pos_gid = (float3) {
            positions[p_idx * 3 + 0],
            positions[p_idx * 3 + 1],
            positions[p_idx * 3 + 2],
            };
        float mass_gid = mass[p_idx];

        int node_idx = num_of_bodies + num_of_nodes; // Root index




        p_idx += get_global_size(0);
    }


}

__kernel void calculate_force_local(
    __global float* positions,
    __global float* mass,
    __global float* vels,
    __global float* accs,
    __global int* children,
    const float theta,
    const float eps,
    const int num_of_bodies,
    const int num_of_nodes,
    const float G,
    const unsigned dt
)
{
    int g_id = get_global_id(0);


    __local volatile int local_pos[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];
    __local volatile int local_node[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];

    __local volatile float dq[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];


#ifdef DEBUG
// TODO: Look out for thread divergence errors!
    if (g_id == 0){
        DEBUG_PRINT(("----FORCE CALCULATION----\n"));
        DEBUG_PRINT(("Theta: %f\n",theta));
        DEBUG_PRINT(("Max Depth Overall: %d\n",MAX_DEPTH));
//        DEBUG_PRINT(("Max Depth of Current Step: %d\n",max_depth));
        DEBUG_PRINT(("Stack Size: %d\n",MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE));

    }
#endif

// Needed for management of stack
    int base = get_local_id(0) / WARP_SIZE;
    int sBase = base * WARP_SIZE;
    int j = base * MAX_DEPTH;


    for (int body_idx = g_id; body_idx < num_of_bodies; body_idx += get_local_size(0) * get_num_groups(0)){
        DEBUG_PRINT(("[%d] Body Index: %d",g_id,body_idx));

        float3 position = (float3){
            positions[body_idx * 3 + 0],
            positions[body_idx * 3 + 1],
            positions[body_idx * 3 + 2],
            };

        float3 acc = (float3) {0.0f,0.0f,0.0f};

        int depth = j;
        if (sBase == get_local_id(0)){
            local_node[depth] = num_of_nodes;
            local_pos[depth] = 0;
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        while (depth >= j) {
            int top;
            while ((top = local_pos[depth] < 8)){
                int child = children[local_node[depth] * NUMBER_OF_CELLS + top];

                if (sBase == get_local_id(0)){
                    local_pos[depth] = top + 1;
                }

                mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

                if (child >= 0){
                    float3 diff_vec = (float3){positions[child * 3 + 0],positions[child * 3 + 1],positions[child * 3 + 2]} - position;
                    float d = distance(position,(float3){positions[child * 3 + 0],positions[child * 3 + 1],positions[child * 3 + 2]});
                    float d_squared = pow(d,2) + pow(eps,2); // + Softening

                    if ((child < num_of_bodies) || work_group_all(d_squared >= dq[depth])){
                        float r_distance = rsqrt(d_squared);
                        DEBUG_PRINT(("\t\t[%d] R Distance: %f\n\t\t\tDifferenc vector: (%f,%f,%f)",g_id,r_distance,diff_vec.x,diff_vec.y,diff_vec.z));

                        float F = mass[child] * r_distance * r_distance * r_distance;
                        acc.x = diff_vec.x * F;
                        acc.y = diff_vec.y * F;
                        acc.z = diff_vec.z * F;

                        DEBUG_PRINT(("\t\t[%d] Force: %f\n",g_id,F));
                    } else {
                        // Node gets pushed to stack;
                        ++depth;
                        if (sBase == get_local_id(0)){
                            local_node[depth] = child;
                            local_pos[depth] = 0;
                        }
                        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                    }
                } else {
                    depth = max(j, depth - 1);
                }
            }
            --depth;
            acc *= G;

            vels[body_idx * 3 + 0] += vels[body_idx * 3 + 0] * acc.x * dt;
            vels[body_idx * 3 + 1] += vels[body_idx * 3 + 1] * acc.y * dt;
            vels[body_idx * 3 + 2] += vels[body_idx * 3 + 2] * acc.z * dt;

            accs[body_idx * 3 + 0] = acc.x;
            accs[body_idx * 3 + 1] = acc.y;
            accs[body_idx * 3 + 2] = acc.z;
        }
    }

}