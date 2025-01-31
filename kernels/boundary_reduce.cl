#include "common.hcl"
#include "boundary_reduce.hcl"

// Boundary
//TODO: local_extremities only needs 3 float! Refactor!
__kernel void hybrid_reduce_root(
    __global float* p,
    __global float* local_extremities,
    __local float* scratch,
    const unsigned num_of_bodies
    )
{
#ifdef DEBUG
    if (0 == get_global_id(0))
        DEBUG_PRINT(("----ROOT REDUCTION - HYBRID REDUCE---"));
#endif

    int l_id = get_global_id(0);


    float3 p_min = (float3){
        p[(get_global_id(0) + 0) % num_of_bodies],
        p[(get_global_id(0) + 1) % num_of_bodies],
        p[(get_global_id(0) + 2) % num_of_bodies]
        };

    float3 p_max = (float3){
        p[(get_global_id(0) + 0) % num_of_bodies],
        p[(get_global_id(0) + 1) % num_of_bodies],
        p[(get_global_id(0) + 2) % num_of_bodies]
        };

    float3 p_min_next, p_max_next;
    for (int g_id = get_global_id(0) + get_global_size(0); g_id < num_of_bodies; g_id += get_global_size(0)){

        p_min_next = (float3) {
            p[g_id * 6 + 0],
            p[g_id * 6 + 1],
            p[g_id * 6 + 2]
            };

        p_max_next = (float3) {
            p[g_id * 6 + 3],
            p[g_id * 6 + 4],
            p[g_id * 6 + 5]
            };
        //TODO
        float3 temp_min = p_min_next;
        p_min_next = get_min_v3(p_min_next,p_max_next);
        p_max_next = get_max_v3(temp_min,p_max_next);

        p_min = get_min_v3(p_min,p_min_next);
        p_max = get_max_v3(p_max,p_max_next);
    }


    scratch[l_id * 6 + 0] = p_min.x;
    scratch[l_id * 6 + 1] = p_min.y;
    scratch[l_id * 6 + 2] = p_min.z;

    scratch[l_id * 6 + 3] = p_max.x;
    scratch[l_id * 6 + 4] = p_max.y;
    scratch[l_id * 6 + 5] = p_max.z;
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 scratch_min = (float3){scratch[l_id * 6 + 0],scratch[l_id * 6 + 1],scratch[l_id * 6 + 2]};
    float3 scratch_max = (float3){scratch[l_id * 6 + 3],scratch[l_id * 6 + 4],scratch[l_id * 6 + 5]};
    float3 scratch_offset_min, scratch_offset_max;

    for (int offset = get_local_size(0) / 2; offset > 0; offset /=2){
        if(l_id < offset){

            scratch_offset_min = (float3){scratch[(l_id + offset) * 6 + 0],scratch[(l_id + offset) * 6 + 1],scratch[(l_id + offset) * 6 + 2]};
            scratch_offset_max = (float3){scratch[(l_id + offset) * 6 + 3],scratch[(l_id + offset) * 6 + 4],scratch[(l_id + offset) * 6 + 5]};

            scratch_min = get_min_v3(scratch_min,scratch_offset_min);
            scratch_max = get_max_v3(scratch_max,scratch_offset_max);

            // Update min boundary value
            scratch[l_id * 6 + 0] = scratch_min.x;
            scratch[l_id * 6 + 1] = scratch_min.y;
            scratch[l_id * 6 + 2] = scratch_min.z;

            // Update max boundary value
            scratch[l_id * 6 + 3] = scratch_max.x;
            scratch[l_id * 6 + 4] = scratch_max.y;
            scratch[l_id * 6 + 5] = scratch_max.z;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (l_id == 0){
        local_extremities[get_group_id(0) * 6 + 0] = scratch[0];
        local_extremities[get_group_id(0) * 6 + 1] = scratch[1];
        local_extremities[get_group_id(0) * 6 + 2] = scratch[2];

        local_extremities[get_group_id(0) * 6 + 3] = scratch[3];
        local_extremities[get_group_id(0) * 6 + 4] = scratch[4];
        local_extremities[get_group_id(0) * 6 + 5] = scratch[5];
    }
}



__kernel void parallel_reduce_root(
    __global float* local_extremities,
    __global float* extremity,
    __local  float* scratch
){
#ifdef DEBUG
    if (get_global_id(0) == 0)
        DEBUG_PRINT(("----ROOT REDUCTION - PARALLEL REDUCE----"));
#endif


    int l_id = get_local_id(0);
    scratch[l_id * 6 + 0] = local_extremities[l_id * 6 + 0];
    scratch[l_id * 6 + 1] = local_extremities[l_id * 6 + 1];
    scratch[l_id * 6 + 2] = local_extremities[l_id * 6 + 2];

    scratch[l_id * 6 + 3] = local_extremities[l_id * 6 + 3];
    scratch[l_id * 6 + 4] = local_extremities[l_id * 6 + 4];
    scratch[l_id * 6 + 5] = local_extremities[l_id * 6 + 5];

    barrier(CLK_LOCAL_MEM_FENCE);


    float3 scratch_min = (float3){scratch[l_id * 6 + 0],scratch[l_id * 6 + 1],scratch[l_id * 6 + 2]};
    float3 scratch_max = (float3){scratch[l_id * 6 + 3],scratch[l_id * 6 + 4],scratch[l_id * 6 + 5]};
    float3 scratch_offset_min, scratch_offset_max;

    for (int offset = get_local_size(0) / 2; offset > 0; offset /=2 ){
        if(l_id < offset){
            scratch_offset_min = (float3){scratch[(l_id + offset) * 6 + 0],scratch[(l_id + offset) * 6 + 1],scratch[(l_id + offset) * 6 + 2]};
            scratch_offset_max = (float3){scratch[(l_id + offset) * 6 + 3],scratch[(l_id + offset) * 6 + 4],scratch[(l_id + offset) * 6 + 5]};

            scratch_min = get_min_v3(scratch_min,scratch_offset_min);
            scratch_max = get_max_v3(scratch_max,scratch_offset_max);

            // Update min boundary value
            scratch[l_id * 6 + 0] = scratch_min.x;
            scratch[l_id * 6 + 1] = scratch_min.y;
            scratch[l_id * 6 + 2] = scratch_min.z;

            // Update max boundary value
            scratch[l_id * 6 + 3] = scratch_max.x;
            scratch[l_id * 6 + 4] = scratch_max.y;
            scratch[l_id * 6 + 5] = scratch_max.z;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id == 0){
        extremity[0] = scratch[0]; // Min. X
        extremity[1] = scratch[1]; // Min. Y
        extremity[2] = scratch[2]; // Min. Z

        extremity[3] = scratch[3]; // Max. X
        extremity[4] = scratch[4]; // Max. Y
        extremity[5] = scratch[5]; // Max. Z
        DEBUG_PRINT(("\tMin. boundary: (%f,%f,%f)\n",extremity[0],extremity[1],extremity[2]));
        DEBUG_PRINT(("\tMax. boundary: (%f,%f,%f)\n",extremity[3],extremity[4],extremity[5]));
    }
}