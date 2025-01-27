#ifndef BOUNDARY_REDUCE_HCL
#define BOUNDARY_REDUCE_HCL

float4 get_min_v4(float4 v1, float4 v2){
    float4 res;
    res.x = min(v1.x,v2.x);
    res.y = min(v1.y,v2.y);
    res.z = min(v1.z,v2.z);
    res.w = min(v1.w,v2.w);

    return res;
}

float4 get_max_v4(float4 v1, float4 v2){
    float4 res;
    res.x = max(v1.x,v2.x);
    res.y = max(v1.y,v2.y);
    res.z = max(v1.z,v2.z);
    res.w = max(v1.w,v2.w);

    return res;
}

float3 get_min_v3(float3 v1, float3 v2){
    float3 res;
    res.x = min(v1.x,v2.x);
    res.y = min(v1.y,v2.y);
    res.z = min(v1.z,v2.z);

    return res;
}

float3 get_max_v3(float3 v1, float3 v2){
    float3 res;
    res.x = max(v1.x,v2.x);
    res.y = max(v1.y,v2.y);
    res.z = max(v1.z,v2.z);

    return res;
}

__kernel void hybrid_reduce_root(
    __global float4* p,
    __global float4* local_extremities,
    __local float4* scratch,
    const unsigned num_of_bodies
);

__kernel void parallel_reduce_root(
    __global float4* local_extremities,
    __global float4* extremity,
    __local  float4* scratch
);

#endif
