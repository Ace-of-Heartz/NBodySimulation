#include "calculate_forces.hcl"
#include "common.hcl"

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
    const float dt,
    __global int *max_depth,
    __global int *errors,
    __global float* boundaries, // Min & Max
    const NumericalMethod method
)
{
    int g_id = get_global_id(0);
    int l_id = get_local_id(0);


    __local volatile int local_pos[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];
    __local volatile int local_node[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];

    __local volatile float dq[MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE];


#ifdef DEBUG
// TODO: Look out for thread divergence errors!
    if (g_id == 0){
        DEBUG_PRINT(("----FORCE CALCULATION----\n"));
        DEBUG_PRINT(("Theta: %f\n",theta));
        DEBUG_PRINT(("Max Depth Overall: %d\n",MAX_DEPTH));
//        DEBUG_PRINT(("Max Depth of Current Step: %d\n",*max_depth));
        DEBUG_PRINT(("Stack Size: %d\n",MAX_DEPTH * WORKGROUP_SIZE / WARP_SIZE));

    }
#endif

    // Select the greatest radius in one direction
    float radius = 0.5f * fmax(fmax(boundaries[3] - boundaries[0], boundaries[4] - boundaries[1]),boundaries[5] - boundaries[2]) ;

    if (theta > 0){
        dq[0] = radius * radius / theta;
    } else {
        dq[0] = radius * radius;
    }

    int i = 0;
    // Set thresholds for each depth
    for (i = 1; i < *max_depth; ++i)
    {
        dq[i] = 0.25f * dq[i - 1];
        dq[i - 1] += eps; // Interesting . . .
    }

    dq[i - 1] += eps;

    if ( *max_depth > MAX_DEPTH){
        DEBUG_PRINT(("ERROR: Max depth error!\n\tExpected: max_depth > %d\n\tActual: %d ",MAX_DEPTH,*max_depth));
        errors[0] = 1;
        return;
    }

    if( *max_depth <= MAX_DEPTH){
        int base = get_local_id(0) / WARP_SIZE;
        int sBase = base * WARP_SIZE;
        int j = base * MAX_DEPTH;

        // Index in Warp?
        int diff = l_id - sBase;

        if (diff < MAX_DEPTH){ // Multiple copies to avoid index calculations later
            dq[diff + j] = dq[diff];
        }

//        int counter = get_local_id(0) % WARP_SIZE;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int body_idx = g_id; body_idx < num_of_bodies; body_idx += get_local_size(0) * get_num_groups(0)){
            DEBUG_PRINT(("[%d] Body Index: %d",g_id,body_idx));

            float3 position = (float3){
                positions[body_idx * 3 + 0],
                positions[body_idx * 3 + 1],
                positions[body_idx * 3 + 2],
                };

            float3 acc = (float3) {0.0f,0.0f,0.0f};

            int depth = j;

            // Stack initialization
            if (l_id == sBase){
                local_node[depth] = num_of_nodes + num_of_bodies;
                local_pos[depth] = 0;
            }

            mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

            while (depth >= j) {
                int top;
                while ((top = local_pos[depth]) < NUMBER_OF_CELLS){
                    int child = children[local_node[depth] * NUMBER_OF_CELLS + top];

                    if (l_id == sBase){
                        local_pos[depth] = top + 1;
                    }

                    mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

                    if (child >= 0){
                        float3 diff_vec = (float3){positions[child * 3 + 0],positions[child * 3 + 1],positions[child * 3 + 2]} - position;

                        float d_squared = eps * eps + diff_vec.x * diff_vec.x + diff_vec.y * diff_vec.y + diff_vec.z * diff_vec.z;

                        if ((child < num_of_bodies) || work_group_all(d_squared >= dq[depth])){ // If body or greater distance than previously set threshold
                            float r_distance = rsqrt(d_squared);
                            DEBUG_PRINT(("\t\t[%d] R Distance: %f\n\t\t\tDifferenc vector: (%f,%f,%f)",g_id,r_distance,diff_vec.x,diff_vec.y,diff_vec.z));

                            float F = mass[child] * r_distance * r_distance * r_distance;
                            acc.x += diff_vec.x * F;
                            acc.y += diff_vec.y * F;
                            acc.z += diff_vec.z * F;

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
            }

            acc *= G;
            // TODO: Change this when "sort" gets implemented

//
//            vels[body_idx * 3 + 0] = vels[body_idx * 3 + 0] + acc.x * dt;
//            vels[body_idx * 3 + 1] = vels[body_idx * 3 + 1] + acc.y * dt;
//            vels[body_idx * 3 + 2] = vels[body_idx * 3 + 2] + acc.z * dt;
//
//            positions[body_idx * 3 + 0] = positions[body_idx * 3 + 0] + vels[body_idx * 3 + 0] * dt;
//            positions[body_idx * 3 + 1] = positions[body_idx * 3 + 1] + vels[body_idx * 3 + 1] * dt;
//            positions[body_idx * 3 + 2] = positions[body_idx * 3 + 2] + vels[body_idx * 3 + 2] * dt;
//            accs[body_idx * 3 + 0] = acc.x;
//            accs[body_idx * 3 + 1] = acc.y;
//            accs[body_idx * 3 + 2] = acc.z;

            switch(method){

                case Leapfrog:
                {
                    float v_i_half_X, v_i_half_Y, v_i_half_Z;
                    v_i_half_X = vels[g_id * 3 + 0] + accs[g_id * 3 + 0] * dt * 0.5f;
                    v_i_half_Y = vels[g_id * 3 + 1] + accs[g_id * 3 + 1] * dt * 0.5f;
                    v_i_half_Z = vels[g_id * 3 + 2] + accs[g_id * 3 + 2] * dt * 0.5f;

                    positions[g_id * 3 + 0] = positions[g_id * 3 + 0] + v_i_half_X * dt;
                    positions[g_id * 3 + 1] = positions[g_id * 3 + 1] + v_i_half_Y * dt;
                    positions[g_id * 3 + 2] = positions[g_id * 3 + 2] + v_i_half_Z * dt;

                    vels[g_id * 3 + 0] = v_i_half_X + acc.x * dt * 0.5f;
                    vels[g_id * 3 + 1] = v_i_half_Y + acc.y * dt * 0.5f;
                    vels[g_id * 3 + 2] = v_i_half_Z + acc.z * dt * 0.5f;

                } break;
                case Euler:
                {
                    accs[g_id * 3 + 0] = acc.x;
                    accs[g_id * 3 + 1] = acc.y;
                    accs[g_id * 3 + 2] = acc.z;

                    vels[g_id * 3 + 0] = vels[g_id * 3 + 0] + acc.x * dt;
                    vels[g_id * 3 + 1] = vels[g_id * 3 + 1] + acc.y * dt;
                    vels[g_id * 3 + 2] = vels[g_id * 3 + 2] + acc.z * dt;

                    positions[g_id * 3 + 0] = positions[g_id * 3 + 0] + vels[g_id * 3 + 0] * dt;
                    positions[g_id * 3 + 1] = positions[g_id * 3 + 1] + vels[g_id * 3 + 1] * dt;
                    positions[g_id * 3 + 2] = positions[g_id * 3 + 2] + vels[g_id * 3 + 2] * dt;
                } break;
            }
        }
    }
}