#include "common.hcl"

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void copy_vertices (
    __global float *p,
    __global float4 *v,
    const unsigned num_of_bodies
    )
{
    int g_id = get_global_id(0);
    int g_size = get_global_size(0);

    for (int i = g_id; i < num_of_bodies; i += g_size){

        //DEBUG_PRINT(("[%d] Body:\n\tID: %d\n\tPosition: (%f,%f,%f)\n\tMass:%f",g_id,i,p[i].x,p[i].y,p[i].z,p[i].w));
        v[i] = (float4){p[i * 3 + 0],p[i * 3 + 1].y,p[i * 3 + 2].z,1.0};
    }

}