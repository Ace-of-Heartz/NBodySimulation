
__constant float distanceThreshold = 0.1f;




float4 get_min(float4 v1, float4 v2){
    float4 res;
    res.x = min(v1.x,v2.x);
    res.y = min(v1.y,v2.y);
    res.z = min(v1.z,v2.z);
    res.w = min(v1.w,v2.w);

    return res;
}

float4 get_max(float4 v1, float4 v2){
    float4 res;
    res.x = max(v1.x,v2.x);
    res.y = max(v1.y,v2.y);
    res.z = max(v1.z,v2.z);
    res.w = max(v1.w,v2.w);

    return res;
}


__kernel void hybrid_reduce_root(
    __global float4* p,
    __global float4* local_extremities,
    __local float4* scratch,
    const unsigned length
)
{
    int l_id = get_global_id(0);
    float4 p_min = p[get_global_id(0) % length];
    float4 p_max = p[get_global_id(0) % length];

    for (int g_id = get_global_id(0) + get_global_size(0); g_id < length; g_id += get_global_size(0)){
        p_min = get_min(p_min,p[g_id]);
        p_max = get_max(p_max,p[g_id]);

    }
    scratch[l_id * 2] = p_min;
    scratch[l_id * 2 + 1] = p_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) / 2; offset > 0; offset /=2){
        if(l_id < offset){
            scratch[l_id * 2] = get_min(scratch[l_id * 2],scratch[l_id * 2 + offset]);
            scratch[l_id * 2 + 1] = get_max(scratch[l_id * 2 + 1],scratch[l_id * 2 + 1 + offset]);
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
            scratch[l_id * 2] = get_min(scratch[l_id * 2],scratch[l_id * 2 + offset]);
            scratch[l_id * 2 + 1] = get_max(scratch[l_id  * 2 + 1],scratch[l_id * 2 + 1+ offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id == 0){
        extremity[0] = scratch[0];
        extremity[1] = scratch[1];
    }
}

int getNthChild(int parentIdx, int nth){

    return parentIdx * 8 + nth;
}

int getParent(int childIdx){
    return (childIdx - childIdx % 8) / 8;
}


int calculateOctantIdx(float4 position, float4 boundaryMin, float4 boundaryMax)
{
    float4 center =  (boundaryMax + boundaryMin) / 2.0f;

    float4 diffVec = position - center;

    int idx = 0;

    idx = 2 - 2 * diffVec.y/fabs(diffVec.y); // y < 0 ? 4 : 0

    idx += 1 - 1 * diffVec.x/fabs(diffVec.x); // x < 0 ? 2 : 0

    idx += 0.5f - 0.5f * diffVec.z/fabs(diffVec.z); // z < 0 ? 1 : 0

    if (diffVec.x > 0){
        boundaryMin.x = position.x;
    }
    else {
        boundaryMax.x = position.x;
    }

    return idx;
}





#define LOCKED -2
#define UNLOCKED 0

#define COMPACT_TREE_BUILD
/*
Build and calculate the nodes of the octree in one kernel using atomic operations
May not as be efficient as using locks and separating building and calculating the tree nodes into two steps!
*/


#ifdef COMPACT_TREE_BUILD
__kernel void build_tree(
    __global float4 *p,
    __global float *m,
    __global float4 *boundaries, //Root boundary max and min
    __global float4 *nodes, // Tree representing the nodes
    __local int children, //Locks for each node in the tree
    unsigned int length,
    unsigned int max_tree_capacity,
    unsigned int max_depth
    )
{
    int l_id = get_local_id(0);
    int g_id = get_global_id(0);

    //Cache local variables
    __local float4 minBoundary, maxBoundary;
    __local float mass;
    __local float4 position;


    minBoundary = boundaries[0];
    maxBoundary = boundaries[1];

    if (length - 1 <=  g_id){
        mass = 0.0f;
        position = (float4)(0.0f,0.0f,0.0f,0.0f);
    } else {
        mass = m[g_id];
        position = p[g_id];
    }

    for (int g_id = get_global_id(0); g_id < max_tree_capacity; g_id += get_global_size(0)){
        locks[g_id] = UNLOCKED; //Initialize locks
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    bool success = false;

    int currentNodeIdx, nextNodeIdx, depth;

    currentNodeIdx = 0; // Root
    depth = 0;


    while(!success && depth < max_depth){
        int octant = calculateOctantIdx(position,minBoundary,maxBoundary);
        adjustBoundaryValues(position,&minBoundary,&maxBoundary);

        nextNodeIdx = getNthChild(octant,nextIdx);


        if (locks[nextNodeIdx] != LOCKED){
            if(atomic_cmpxchg(&locks[nextNodeIdx],UNLOCKED,LOCKED) == UNLOCKED )
            {
                // Locking succeed
                if (nodes[nextNodeIdx]){
                    // Cell was NOT null
                    int original_octant, conflict_octant;
                    int original_idx, conflict_idx;
                    float4 original = nodes[nextNodeIdx];
                    do {

                        original_octant = calculateOctantIdx(original,minBoundary,maxBoundary);
                        original_idx = getNthChild(currentNodeIdx,original_octant);

                        conflict_octant = calculateOctantIdx(position,minBoundary,maxBoundary);
                        conflict_idx = getNthChild(currentNodeIdx,conflict_idx);

                        currentNodeIdx = original_idx;
                        ++depth;
                    } while(original_octant != conflict_octant && depth < max_depth)

                    if (depth < max_depth){
                        nodes[original_idx] = (float4)(original.x,original.y,original.z,original.w);
                        nodes[conflict_idx] = (float4)(position.x,position.y,position.z,mass);
                    } else { // Treat the particles as one particle
                        nodes[original_idx] = (float4)(original.x,original.y,original.z,original.w);
                        notes[conflict_idx] += (float4)(position.x,position.y,position.z,mass);

                    }

                }
                else {
                    // Cell was null
                    nodes[nextNodeIdx] = (float4)(position.x,position.y,position.z,mass);

                    barrier(CLK_LOCAL_MEM_FENCE);
                    atomic_xchg(&locks[nextNodeIdx],LEAF);
                }
                success = true;
            }


            //Processed or locked
        } else { //LOCKED


        }

        currentNodeIdx = nextNodeIdx;
        ++depth;
    }




}

#else
__kernel void build_tree(

)
#endif


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

