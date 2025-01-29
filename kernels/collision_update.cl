#include "common.hcl"

float3 calc_collision_traj(float3 p_i, float3 p_j,float r_i, float r_j,float3 v_i, float3 v_j) {
    //For now, let all bodies have 1 radius length
    //TODO
    float3 diff_vec = p_j - p_i;

    float3 norm = normalize(diff_vec); // Assume the body is a sphere (and maybe a cow)


    if (distance(p_i,p_j) > (r_i + r_j))
        return v_i;
    else
        return (v_i - 2.0f * dot(norm,v_i) * norm);
}

__kernel void update_collision_local(
	__global float *v, // state: speed
	__global float *p, // state: position and mass
	const int num_of_bodies
//	__global float *r,
		)
{
#ifdef DEBUG
    if (0 == get_global_id(0))
        DEBUG_PRINT(("---BRUTE FORCE UPDATE - LOCAL---\n\tdt: %f\ntNumber of bodies: %d\n\tGlobal size: %d",dt,num_of_bodies,get_global_size(0)));
#endif

	int g_id = get_global_id(0);
	int l_id = get_local_id(0);


	// Aggregate acceleration for each object

    __local float shPos[WORKGROUP_SIZE * 7];

    float3 p_gid;
    float3 vel_gid;
    if(g_id < num_of_bodies){ // Only the last workgroup will have thread divergence here
        p_gid = (float3){
            p[g_id * 3 + 0],
            p[g_id * 3 + 1],
            p[g_id * 3 + 2],
        };

        vel_gid = (float3) {
            v[g_id * 3 + 0],
            v[g_id * 3 + 1],
            v[g_id * 3 + 2]
        };
    }
    int tile, idx;

    for (tile = 0, idx = tile * WORKGROUP_SIZE + l_id; idx < num_of_bodies; ++tile){

        shPos[l_id * 7 + 0] = p[idx * 3 + 0];
        shPos[l_id * 7 + 1] = p[idx * 3 + 1];
        shPos[l_id * 7 + 2] = p[idx * 3 + 2];
        shPos[l_id * 7 + 3] = v[idx * 3 + 0];
        shPos[l_id * 7 + 4] = v[idx * 3 + 1];
        shPos[l_id * 7 + 5] = v[idx * 3 + 2];
        shPos[l_id * 7 + 6] = 0.010f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int j = 0; j < WORKGROUP_SIZE; ++j){
            if(g_id == tile * WORKGROUP_SIZE + j)
                continue;
            vel_gid = calc_collision_traj(p_gid,(float3){shPos[j * 7 + 0],shPos[j * 7 + 1],shPos[j * 7 + 2]},0.010f,shPos[j * 7 + 6],vel_gid,(float3){shPos[j * 7 + 3],shPos[j * 7 + 4],shPos[j * 7 + 5]});

        }
        barrier(CLK_LOCAL_MEM_FENCE);
        idx = tile * WORKGROUP_SIZE + l_id;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if(g_id < num_of_bodies) // Only the last workgroup will have thread divergence here
    {
        v[g_id * 3 + 0] = vel_gid.x;
        v[g_id * 3 + 1] = vel_gid.y;
        v[g_id * 3 + 2] = vel_gid.z;

    }


}