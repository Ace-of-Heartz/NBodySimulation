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
    int l_id = get_global_id(0);




    float3 p_min = (float3)p[(get_global_id(0) + 0) % num_of_bodies];
    float3 p_max = (float3)p[get_global_id(0) % num_of_bodies];

    for (int g_id = get_global_id(0) + get_global_size(0); g_id < num_of_bodies; g_id += get_global_size(0)){
        p_min = get_min_v4(p_min,p[g_id]);
        p_max = get_max_v4(p_max,p[g_id]);

    }
    scratch[l_id * 2] = p_min;
    scratch[l_id * 2 + 1] = p_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) / 2; offset > 0; offset /=2){
        if(l_id < offset){
            scratch[l_id * 2] = get_min_v4(scratch[l_id * 2],scratch[l_id * 2 + offset]);
            scratch[l_id * 2 + 1] = get_max_v4(scratch[l_id * 2 + 1],scratch[l_id * 2 + 1 + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (l_id == 0){
        local_extremities[get_group_id(0) * 2] = scratch[0];
        local_extremities[get_group_id(0) * 2 + 1] = scratch[1];
    }
}

__kernel void parallel_reduce_root(
    __global float4* local_extremities,
    __global float4* extremity,
    __local  float4* scratch
){
    int l_id = get_local_id(0);
    scratch[l_id * 2] = local_extremities[l_id * 2];
    scratch[l_id * 2 + 1] = local_extremities[l_id * 2 + 1];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) / 2; offset > 0; offset /=2 ){
        if(l_id < offset){
            scratch[l_id * 2] = get_min_v4(scratch[l_id * 2],scratch[l_id * 2 + offset]);
            scratch[l_id * 2 + 1] = get_max_v4(scratch[l_id  * 2 + 1],scratch[l_id * 2 + 1+ offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id == 0){
        extremity[0] = scratch[0];
        extremity[1] = scratch[1];
        DEBUG_PRINT(("Min. boundary: (%f,%f,%f)\n",extremity[0].x,extremity[0].y,extremity[0].z));
        DEBUG_PRINT(("Max. boundary: (%f,%f,%f)\n",extremity[1].x,extremity[1].y,extremity[1].z));
    }
}