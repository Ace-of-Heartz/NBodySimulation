

__kernel void update(
	__global float *v, // state: speed
	__global float *p, // state: position
	__global float* m,  // mass
	float dt)		   // delta time (time between frames)
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

	const float G = 0.0001f; // gravitational constant

	int gid = get_global_id(0);
	float3 p_gid = vload3(gid,p);
	float m_gid = m[gid];

	// Aggregate acceleration for each object
	float3 a_gid = {0.0f, 0.0f,0.0f};

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
		float3 p_other = vload3(i,p);
		float m_other = m[i];


        float d = distance(p_gid,p_other);
        float3 d_vec = normalize(p_other - p_gid);
        float eps = 0.01;
        a_gid += (m_other) / pow((pow(d,2) + pow(eps,2)),3/2) * d_vec;

		// 1. compute force and distance
		// 2. compute acceleration
		// 3. aggregate acceleration
	}
	a_gid *= G;

	// 4. integrate speed (using dt and a_gid)
	// TODO: Use Shared Memory for saving previous acceleration
	float3 res = v[gid] + dt * (a_gid);
//	float3 res = {1.0f,1.0f,1.0f};
    vstore3(res,gid,v);

	// 5. integrate position
	res = p[gid] + dt * v[gid];
    vstore3(res,gid,p);
}

