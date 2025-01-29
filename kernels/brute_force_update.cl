#include "common.hcl"

__kernel void update(
	__global float *v, // state: speed
	__global float *p, // state: position and mass
	__global float *m,
	__global float *a, // acceleration
	float dt, // delta time (time between frames)
	float G // gravitational constant
	)
{
#ifdef DEBUG
    if (0 == get_global_id(0))
        DEBUG_PRINT(("---BRUTE FORCE UPDATE---\n\tdt: %f\n",dt));
#endif

	// Numerical integration for speed and distance
	//   v = a * t
	//   s = v * t

	// Relate force, mass and acceleration
	//   F = m*a

	// Newton: Law of universal gravitation
	//   Given two objects with mass m_1, m_2 and distance d
	//   F_grav = (G * m_1 * m_2) / d^2
	//
	// (Tip: add a small constant to divisor to avoid numerical errors!)

	int gid = get_global_id(0);
    float3 p_gid = (float3){
        p[gid * 3 + 0],
        p[gid * 3 + 1],
        p[gid * 3 + 2],
    };

    float mass = m[gid];

	// Aggregate acceleration for each object
	float3 a_gid = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
        float3 p_other = (float3){
            p[i * 3 + 0],
            p[i * 3 + 1],
            p[i * 3 + 2],
        };

        float m_other = m[i];

        float d = distance(p_gid,p_other);
        float3 d_vec = normalize(p_other - p_gid);
        float eps = .01;
        float dist_sqr = d_vec.x * d_vec.x + d_vec.y * d_vec.y + d_vec.z * d_vec.z + eps * eps;
        float s = m_other / pow((dist_sqr),3/2);

        a_gid += d_vec * s;
//        a_gid += ((m_other) / pow((pow(d,2) + pow(eps,2)),3/2)) * d_vec;

		// 1. compute force and distance
		// 2. compute acceleration
		// 3. aggregate acceleration
	}
	a_gid *= G;

	// 4. integrate speed (using dt and a_gid)

    barrier(CLK_GLOBAL_MEM_FENCE);

    a[gid + 3 + 0] = a_gid.x;
    a[gid + 3 + 1] = a_gid.y;
    a[gid + 3 + 2] = a_gid.z;
	// 5. integrate position

    v[gid * 3 + 0] = v[gid * 3 + 0] + a_gid.x * dt;
    v[gid * 3 + 1] = v[gid * 3 + 1] + a_gid.y * dt;
    v[gid * 3 + 2] = v[gid * 3 + 2] + a_gid.z * dt;

    p[gid * 3 + 0] = p[gid * 3 + 0] + v[gid * 3 + 0] * dt;
    p[gid * 3 + 1] = p[gid * 3 + 1] + v[gid * 3 + 1] * dt;
    p[gid * 3 + 2] = p[gid * 3 + 2] + v[gid * 3 + 2] * dt;

}


float3 calc_body_body(
    float3 pos_i, float m_i,
    float3 pos_j, float m_j,
    float3 acc_i
    )
{
    float d = distance(pos_i,pos_j);
    float3 d_vec = normalize(pos_j - pos_i);
    float eps = 0.1f;

    float dist_sqr = d_vec.x * d_vec.x + d_vec.y * d_vec.y + d_vec.z * d_vec.z + eps * eps;

    float s = m_j / pow((dist_sqr),3/2);

    acc_i += d_vec * s;
    return acc_i;
}



__kernel void update_local(
	__global float *v, // state: speed
	__global float *p, // state: position and mass
	__global float *m,
	__global float *a, // acceleration
	const float dt, // delta time (time between frames)
	const float G, // gravitational constant
	const int num_of_bodies
		)
{
#ifdef DEBUG
    if (0 == get_global_id(0))
        DEBUG_PRINT(("---BRUTE FORCE UPDATE - LOCAL---\n\tdt: %f\ntNumber of bodies: %d\n\tGlobal size: %d",dt,num_of_bodies,get_global_size(0)));
#endif

	int g_id = get_global_id(0);
	int l_id = get_local_id(0);


	// Aggregate acceleration for each object

    __local float4 shPos[WORKGROUP_SIZE];

    float mass;
    float3 p_gid;
    float3 a_gid;
    if(g_id < num_of_bodies){ // Only the last workgroup will have thread divergence here
        p_gid = (float3){
            p[g_id * 3 + 0],
            p[g_id * 3 + 1],
            p[g_id * 3 + 2],
        };

        mass = m[g_id];
    }
    int tile, idx;
    a_gid = (float3){0.0f, 0.0f, 0.0f};

    for (tile = 0, idx = tile * WORKGROUP_SIZE + l_id; idx < num_of_bodies; ++tile){

        shPos[l_id] = (float4){
            p[(idx) * 3 + 0],
            p[(idx) * 3 + 1],
            p[(idx) * 3 + 2],
            m[(idx)]
        };

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int j = 0; j < WORKGROUP_SIZE; ++j){
            a_gid = calc_body_body(p_gid,mass,(float3){shPos[j].x,shPos[j].y,shPos[j].z},shPos[j].w,a_gid);

        }
        barrier(CLK_LOCAL_MEM_FENCE);
        idx = tile * WORKGROUP_SIZE + l_id;
    }
    a_gid *= G;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if(g_id < num_of_bodies) // Only the last workgroup will have thread divergence here
    {
        a[g_id + 3 + 0] = a_gid.x;
        a[g_id + 3 + 1] = a_gid.y;
        a[g_id + 3 + 2] = a_gid.z;

        v[g_id * 3 + 0] = v[g_id * 3 + 0] + a_gid.x * dt;
        v[g_id * 3 + 1] = v[g_id * 3 + 1] + a_gid.y * dt;
        v[g_id * 3 + 2] = v[g_id * 3 + 2] + a_gid.z * dt;

        p[g_id * 3 + 0] = p[g_id * 3 + 0] + v[g_id * 3 + 0] * dt;
        p[g_id * 3 + 1] = p[g_id * 3 + 1] + v[g_id * 3 + 1] * dt;
        p[g_id * 3 + 2] = p[g_id * 3 + 2] + v[g_id * 3 + 2] * dt;

        g_id += get_global_size(0);
    }


}
