

__kernel void update(
	__global float2 *v, // state: speed
	__global float2 *p, // state: position
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
	float2 p_gid = p[gid];
	float m_gid = m[gid];

	// Aggregate acceleration for each object
	float2 a_gid = {0.0f, 0.0f};

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
		float2 p_other = p[i];
		float m_other = m[i];


        float d = distance(p_gid,p_other);
        float2 d_vec = normalize(p_other - p_gid);
        float eps = 0.01;
        a_gid += (m_other) / pow((pow(d,2) + pow(eps,2)),3/2) * d_vec;

		// 1. compute force and distance
		// 2. compute acceleration
		// 3. aggregate acceleration
	}
	a_gid *= G;

	// 4. integrate speed (using dt and a_gid)
	// TODO: Use Shared Memory for saving previous acceleration
	v[gid] = v[gid] + dt * (a_gid);

	// 5. integrate position
	p[gid] = p[gid] + dt * v[gid];
}

