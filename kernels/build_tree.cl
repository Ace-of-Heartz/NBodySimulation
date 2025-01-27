#include "common.hcl"
#include "build_tree.hcl"

__kernel void build_tree(
    __global float *positions,   // Node and body positions
    __global float *mass,
    __global float *boundaries,  // Root boundary max and min
    __global int *children,      // Array holding indexes to bodies and nodes
    const unsigned max_depth,
    const unsigned max_children, // Size of children
    const int num_of_bodies,
    const int num_of_nodes,
    __global volatile int* bottom,
    __global volatile int* error
    )
{
    int l_id = get_local_id(0);
    int g_id = get_global_id(0);

#ifdef DEBUG
    if (g_id == 0){
        DEBUG_PRINT(("----BUILD TREE KERNEL----\nMax Depth: %d\nMax children: %d\nNumber of bodies: %d\nNumber of Nodes: %d\n",max_depth,max_children,num_of_bodies,num_of_nodes));
    }
#endif
    //Cache local variables
    float3 boundaryMin, boundaryMax;
    float massIdx;
    float3 position;

    for (int i = g_id; i < max_children; i += get_global_size(0)){
        children[i] = DEFAULT; // Set all children to default value
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    int p_idx = g_id;
    int child_path = 0;
    int child_idx = 0;
    int node_idx = 0;

    bool new_body = true;

    while (p_idx < num_of_bodies)
    {

        if (new_body)
        {
            position = (float3){
                positions[p_idx * 3 + 0],
                positions[p_idx * 3 + 1],
                positions[p_idx * 3 + 2],
                };
            massIdx = mass[p_idx];
            boundaryMin = (float3){boundaries[0],boundaries[1],boundaries[2]};
            boundaryMax = (float3){boundaries[3],boundaries[4],boundaries[5]};

            child_path = 0;

            new_body = false;

            DEBUG_PRINT(("\t[%d] Processing new body:\n\t\tIndex: %d\n\t\tPosition: (%f,%f,%f)\n",g_id,p_idx,position.x,position.y,position.z));
        }

        child_idx = children[child_path];

        while(child_idx >= num_of_bodies) // Traverse the tree until we find an empty node (DEFAULT index) or a leaf (index of body)
        {
            node_idx = child_idx;
            child_path = calculateOctantIdx(position,boundaryMin,boundaryMax);

            child_idx = children[node_idx * NUMBER_OF_CELLS + child_path];
            adjustBoundaryValues(position,&boundaryMin,&boundaryMax);
//            DEBUG_PRINT(("[%d] Path: %d\nIdx: %d\n",g_id,child_path + node_idx * NUMBER_OF_CELLS ,child_idx));
        }

        if (child_idx != LOCKED){

            int locked = node_idx * NUMBER_OF_CELLS + child_path;
            if (atomic_cmpxchg(&children[locked],child_idx,LOCKED) == child_idx)
                if (child_idx == DEFAULT){
                    // Empty node -> Insert body directly
                    DEBUG_PRINT(("\t[%d] Empty node: %d parent - %d child\n",g_id,node_idx,child_path + node_idx * NUMBER_OF_CELLS));

                    children[locked] = p_idx;
                }
                else
                {
                    // Occupied node -> Create subtree with original and current body
                    DEBUG_PRINT(("\t[%d] Non-empty node:\n\tPointing to: %d\n\tParent: %d\n",g_id,child_idx,node_idx));
                    int patch = -1;
                    float3 og_position = (float3){
                        positions[child_idx * 3 + 0],
                        positions[child_idx * 3 + 1],
                        positions[child_idx * 3 + 2],
                        };
                    float3 og_boundary_min,og_boundary_max;
                    og_boundary_min = boundaryMin;
                    og_boundary_max = boundaryMax;

                    do {
                        const int cell = atom_dec(bottom) - 1;
                        if (cell <= num_of_bodies){
//                            DEBUG_PRINT(("\t\t[%d] ERROR: Cell capacity overflow: %d",g_id,cell));
                            *error = 1;
                            *bottom = num_of_nodes;
                            return;
                        }

                        patch = max(cell,patch);

//                        DEBUG_PRINT(("\t\t[%d] Patch & Cell: %d,%d",g_id,patch,cell));

                        positions[cell * 3 + 0] = (boundaryMax.x + boundaryMin.x) / 2.0f;
                        positions[cell * 3 + 1] = (boundaryMax.y + boundaryMin.y) / 2.0f;
                        positions[cell * 3 + 2] = (boundaryMax.z + boundaryMin.z) / 2.0f;
                        mass[cell] = -1.0f;

                        // Set child nodes to default value
                        for (int k = 0; k < NUMBER_OF_CELLS; k++) {
                            children[cell * NUMBER_OF_CELLS + k] = DEFAULT;
//                            DEBUG_PRINT(("\t\t[%d] Child %d set to default value. Maximum number of children: %d\n",g_id,cell * NUMBER_OF_CELLS + k,max_children));

                        }

                        if (patch != cell) {
                            children[node_idx * NUMBER_OF_CELLS + child_path] = cell; //Insert node
                        }


                        child_path = calculateOctantIdx(og_position,og_boundary_min,og_boundary_max);
                        adjustBoundaryValues(og_position,&og_boundary_min,&og_boundary_max);

//                        DEBUG_PRINT(("[%d] Insert original body %d at: %d ",g_id,child_idx,NUMBER_OF_CELLS * node_idx + child_path));
                        children[NUMBER_OF_CELLS * cell + child_path] = child_idx; //Original body index

                        node_idx = cell;
                        child_path = calculateOctantIdx(position,boundaryMin,boundaryMax);
                        adjustBoundaryValues(position,&boundaryMin,&boundaryMax);

                        child_idx = children[NUMBER_OF_CELLS * node_idx + child_path];


                    } while (child_idx >= 0);
//                    DEBUG_PRINT(("[%d] Insert body %d at: %d ",g_id,p_idx,NUMBER_OF_CELLS * node_idx + child_path));


                    children[NUMBER_OF_CELLS * node_idx + child_path] = p_idx;
                    children[locked] = patch;

                }

                p_idx = get_global_size(0);
                new_body = true;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }



}