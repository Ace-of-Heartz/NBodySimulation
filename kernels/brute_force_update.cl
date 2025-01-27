__kernel void update(
	__global float *v, // state: speed
	__global float4 *p_m, // state: position and mass
	__global float *a, // acceleration
	float dt, // delta time (time between frames)
	float G // gravitational constant
	)
{

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
        p_m[gid].x,
        p_m[gid].y,
        p_m[gid].z
        };

    float mass = p_m[gid].w;

	// Aggregate acceleration for each object
	float3 a_gid = {0.0f, 0.0f,0.0f};

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
        float3 p_other = (float3){
            p_m[i].x,
            p_m[i].y,
            p_m[i].z
            };
        float m_other = p_m[i].w;

        float d = distance(p_gid,p_other);
        float3 d_vec = normalize(p_other - p_gid);
        float eps = .01;
        a_gid += ((m_other) / pow((pow(d,2) + pow(eps,2)),3/2)) * d_vec;

		// 1. compute force and distance
		// 2. compute acceleration
		// 3. aggregate acceleration
	}
	a_gid *= G;

	// 4. integrate speed (using dt and a_gid)
	// TODO: Use Shared Memory for saving previous acceleration


    // Leapfrog!
//    float4 v_gid_half = v[gid] + a[gid] * dt / 2.0f;
//    p[gid] = p[gid] + v_gid_half * dt;
//    v[gid] = v_gid_half + a_gid * dt;
//    a[gid] = a_gid;
	// 5. integrate position
    v[gid * 3] = v[gid * 3] + a_gid.x * dt;
    v[gid * 3 + 1] = v[gid * 3 + 1] + a_gid.y * dt;
    v[gid * 3 + 2] = v[gid * 3 + 2] + a_gid.z * dt;

//	v[gid] = v[gid] + a_gid * dt;
//    p[gid] = p[gid] + v[gid] * dt;

    p_m[gid] = (float4){
        p_m[gid].x + v[gid * 3] * dt,
        p_m[gid].y + v[gid * 3 + 1] * dt,
        p_m[gid].z + v[gid * 3 + 2] * dt,
        p_m[gid].w};
}

struct Body{
    float vX;
    float vY;
    float vZ;

    float pX;
    float pY;
    float pZ;

    float M;

    float aX;
    float aY;
    float aZ;
};

__kernel void update_local(
	__global float4 *v, // state: speed
	__global float4 *p, // state: position
	__global float *m,  // mass
	__global float4 *a,
	__local struct Body *obj,
	float dt, // delta time (time between frames)
	float G // gravitational constant
	)
{

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
    float4 p_gid = p[gid];
	float m_gid = m[gid];




	//TODO
//	for (int i = gid; i < num_of_bodies; i += get_global_size()){
//	    obj[gid] =
//
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);

	// Aggregate acceleration for each object
	float4 a_gid = {0.0f, 0.0f,0.0f,0.0f};

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
        float4 p_other = p[i];
		float m_other = m[i];


        float d = distance(p_gid,p_other);
        float4 d_vec = normalize(p_other - p_gid);
        float eps = .01;
        a_gid += ((m_other) / pow((pow(d,2) + pow(eps,2)),3/2)) * d_vec;

		// 1. compute force and distance
		// 2. compute acceleration
		// 3. aggregate acceleration
	}
	a_gid *= G;

	// 4. integrate speed (using dt and a_gid)
	// TODO: Use Shared Memory for saving previous acceleration


    // Leapfrog!
//    float4 v_gid_half = v[gid] + a[gid] * dt / 2.0f;
//    p[gid] = p[gid] + v_gid_half * dt;
//    v[gid] = v_gid_half + a_gid * dt;
//    a[gid] = a_gid;
	// 5. integrate position


	v[gid] = v[gid] + a_gid * dt;
    p[gid] = p[gid] + v[gid] * dt;
}