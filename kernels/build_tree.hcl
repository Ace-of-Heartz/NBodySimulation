#ifndef BUILD_TREE_HCL
#define BUILD_TREE_HCL

int getNthChildIdx(int parentIdx, int nth){

    return parentIdx * 8 + nth;
}

int getParentIdx(int childIdx){
    return (childIdx - childIdx % 8) / 8;
}

// Returns the index of the an octant with `(boundaryMin + boundaryMax) / 2` center.
int calculateOctantIdx(float4 position, float4 boundaryMin, float4 boundaryMax)
{
    float4 center =  (boundaryMax + boundaryMin) / 2.0f;

    float4 diffVec = position - center;

    int idx = 0;

    idx = 2 - 2 * diffVec.y/fabs(diffVec.y); // y < 0 ? 4 : 0

    idx += 1 - 1 * diffVec.x/fabs(diffVec.x); // x < 0 ? 2 : 0

    idx += 0.5f - 0.5f * diffVec.z/fabs(diffVec.z); // z < 0 ? 1 : 0

    return idx;
}

void adjustBoundaryValues(float4 position, float4* boundaryMin, float4* boundaryMax){

    float4 center = (*boundaryMax + *boundaryMin) / 2.0f;
    float4 diffVec = position - center;

    if (diffVec.x > 0){
        boundaryMin->x = center.x;
    } else {
        boundaryMax->x = center.x;
    }

    if (diffVec.x > 0){
        boundaryMin->y = center.y;
    } else {
        boundaryMax->y = center.y;
    }

    if (diffVec.z > 0){
        boundaryMin->z = center.z;
    } else {
        boundaryMax->z = center.z;
    }

}

#endif