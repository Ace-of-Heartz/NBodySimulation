#include "common.hcl"
#include "build_tree.hcl"

__kernel void build_tree(
    __global float4 *bodies_n_nodes,
    __global float4 *boundaries, //Root boundary max and min
    __global int *children, //Locks for each node in the tree
    const unsigned max_depth,
    const unsigned max_children,
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
        DEBUG_PRINT(("----BUILD TREE KERNEL----\nMax Depth: %d\nMax children: %d\nNumber of bodies: %d\nNumber of Nodes:%d\n",max_depth,max_children,num_of_bodies,num_of_nodes));
    }
#endif
    //Cache local variables
    float4 boundaryMin, boundaryMax;
    float mass;
    float4 position;

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
            position = bodies_n_nodes[p_idx];
            boundaryMin = boundaries[0];
            boundaryMax = boundaries[1];
            child_path = 0;

            new_body = false;
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
                    DEBUG_PRINT(("[%d] Empty node: %d parent - %d child\n",g_id,node_idx,child_path + node_idx * NUMBER_OF_CELLS));

                    children[locked] = p_idx;
                }
                else
                {
                    // Occupied node -> Create subtree with original and current body
                    DEBUG_PRINT(("[%d] Non-empty node:\n\tPointing to: %d\n\tParent: %d\n",g_id,child_idx,node_idx));
                    int patch = -1;
                    float4 og_position = bodies_n_nodes[child_idx];
                    float4 og_boundary_min,og_boundary_max;
                    og_boundary_min = boundaryMin;
                    og_boundary_max = boundaryMax;


                    do {
                        const int cell = atom_dec(bottom) - 1;
                        if (cell <= num_of_bodies){
                            DEBUG_PRINT(("[%d] ERROR: Cell capacity overflow: %d",g_id,cell));
                            *error = 1;
                            return;
                        }

                        patch = max(cell,patch);


                        bodies_n_nodes[cell] = (boundaryMax + boundaryMin) / 2.0f;
                        bodies_n_nodes[cell].w = -1.0f;

//                         Set child nodes to default value
                        for (int k = 0; k < NUMBER_OF_CELLS; k++) children[cell * NUMBER_OF_CELLS + k] = DEFAULT;

                        if (patch != cell) {
                            children[node_idx * NUMBER_OF_CELLS + child_path] = cell; //Insert node
                        }


                        child_path = calculateOctantIdx(og_position,og_boundary_min,og_boundary_max);
                        adjustBoundaryValues(og_position,&og_boundary_min,&og_boundary_max);

                        DEBUG_PRINT(("[%d] Insert original body %d at: %d ",child_idx,p_idx,NUMBER_OF_CELLS * node_idx + child_path));
                        children[NUMBER_OF_CELLS * cell + child_path] = child_idx; //Original body index

                        node_idx = cell;
                        child_path = calculateOctantIdx(position,boundaryMin,boundaryMax);
                        adjustBoundaryValues(position,&boundaryMin,&boundaryMax);

                        child_idx = children[NUMBER_OF_CELLS * node_idx + child_path];


                    } while (child_idx >= 0);
                    DEBUG_PRINT(("[%d] Insert body %d at: %d ",g_id,p_idx,NUMBER_OF_CELLS * node_idx + child_path));


                    children[NUMBER_OF_CELLS * node_idx + child_path] = p_idx;
                    children[locked] = patch;

                }

                p_idx = get_global_size(0);
                new_body = true;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }



}