

__kernel void update(
	__global float4 *v, // state: speed
	__global float4 *p, // state: position
	__global float *m,  // mass
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

    v[gid] = v[gid] + a_gid * dt;

	// 5. integrate position
    p[gid] = p[gid] + v[gid] * dt;

}

