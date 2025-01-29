//
// Created by ace on 1/28/25.
//

#ifndef BARNESHUTCONFIG_H
#define BARNESHUTCONFIG_H



class BarnesHutConfig {
public:
    BarnesHutConfig() {};
    ~BarnesHutConfig() {};

    BarnesHutConfig(const BarnesHutConfig& other)
        : max_octree_depth(other.max_octree_depth),
          theta(other.theta),
          epsilon(other.epsilon)
    {
    }

    BarnesHutConfig(BarnesHutConfig&& other) noexcept
        : max_octree_depth(other.max_octree_depth),
          theta(other.theta),
          epsilon(other.epsilon)
    {
    }

    BarnesHutConfig& operator=(const BarnesHutConfig& other)
    {
        if (this == &other)
            return *this;
        max_octree_depth = other.max_octree_depth;
        theta = other.theta;
        epsilon = other.epsilon;
        return *this;
    }

    BarnesHutConfig& operator=(BarnesHutConfig&& other) noexcept
    {
        if (this == &other)
            return *this;
        max_octree_depth = other.max_octree_depth;
        theta = other.theta;
        epsilon = other.epsilon;
        return *this;
    }

    int& GetMaxOctreeDepth() {return max_octree_depth;}
    float& GetTheta() {return theta;}
    float& GetEpsilon() {return epsilon;}

    void SetMaxOctreeDepth(int max_octree_depth){ this->max_octree_depth = max_octree_depth;}
    void SetTheta(float theta){ this->theta = theta;}
    void SetEpsilon(float epsilon){ this->epsilon = epsilon;}

private:
    int max_octree_depth = 32;
    float theta = 1.5f; // Distance criteria
    float epsilon = 0.01f; // Smoothing value
};



#endif //BARNESHUTCONFIG_H
