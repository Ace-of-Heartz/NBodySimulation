__kernel void update(
	__global float *v, // state: speed
	__global float *p, // state: position and mass
	__global float *m,
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

