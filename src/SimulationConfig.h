//
// Created by ace on 2025.01.15..
//

#ifndef SIMULATIONCONFIG_H
#define SIMULATIONCONFIG_H

#include "BarnesHutConfig.h"

enum NumericalMethod
{
    LEAPFROG = 1,
    EULER = 2
};

enum PositionConfig
{
    SPHERE_POS,
    UNIFORM_POS,
};

enum VelocityConfig
{
    RANDOM_VEL,
    STARTING_OUT_VEL,
    STARTING_IN_VEL,
    FUNC_ZERO_VEL,
    TANGENT_XZ_VEL,
};


enum AlgorithmConfig
{
    BARNES_HUT,
    BRUTE_FORCE_GLOBAL, // Naive
    BRUTE_FORCE_LOCAL, // Optimized with local cache TODO

};

struct NormalDistribution
{
    float mean;
    float deviation;
};

class SimulationConfig {

public:
    SimulationConfig() = default;
    SimulationConfig(const SimulationConfig& other);
    SimulationConfig(SimulationConfig&& other) noexcept;

    SimulationConfig& operator=(const SimulationConfig& other);
    SimulationConfig& operator=(SimulationConfig&& other) noexcept;

    int& GetNumberOfBodies();
    float& GetGravitationalConstant();
    float& GetStartingSpeedMul();
    float& GetStartingVolumeRadius();
    int& GetNumberOfMassiveObjects();
    float& GetMassiveObjectMass();
    PositionConfig& GetPositionConfig();
    VelocityConfig& GetVelocityConfig();
    NormalDistribution& GetMassDistribution();
    AlgorithmConfig& GetAlgorithmConfig();
    BarnesHutConfig& GetBarnesHutConfig();
    NumericalMethod& GetNumericalMethod();
    bool& GetCollision();

    void SetNumberOfBodies(int number_of_particles);
    void SetGravitationalConstant(float gravitational_constant);
    void SetStartingSpeedMul(float starting_speed_mul);
    void SetStartingVolumeRadius(float starting_volume_radius);
    void SetNumberOfMassiveObjects(int number_of_massive_objects);
    void SetMassiveObjectMass(float massive_object_mass);
    void SetPositionConfig(PositionConfig pos_config);
    void SetVelocityConfig(VelocityConfig vel_config);
    void SetMassDistribution(const NormalDistribution& mass_distr);
    void SetAlgorithmConfig(AlgorithmConfig algo);
    void SetBarnesHutConfig(BarnesHutConfig barnes_hut_config);
    void SetNumericalMethod(NumericalMethod numerical_method);
    void SetCollision(bool collision);

private:
    int num_of_bodies = 20000;
    float gravitational_constant = 6.67e-11;

    float starting_speed_mul = 1.0f;
    float starting_volume_radius = 1.5f;

    int number_of_massive_objects = 0;
    float massive_object_mass = 1.0f;

    bool collision = false;

    PositionConfig pos_config = UNIFORM_POS;
    VelocityConfig vel_config = FUNC_ZERO_VEL;
    AlgorithmConfig algo_config = BRUTE_FORCE_GLOBAL;

    NormalDistribution mass_distr = NormalDistribution(0.5,0.25);

    BarnesHutConfig barnes_hut_config;

    NumericalMethod numerical_method = EULER;
};

struct Simulation
{
    [[nodiscard]] SimulationConfig GetConfig() const
    {
        return config;
    }

    void SetConfig(const SimulationConfig& config)
    {
        this->config = config;
    }

    [[nodiscard]] float& GetSimulationSpeedMul()
    {
        return simulationSpeedMul;
    }

    [[nodiscard]] float& GetParticleSize()
    {
        return particleSize;
    }
private:
    SimulationConfig config;

    float simulationSpeedMul = 1.0f;
    float particleSize = 0.05f;
};


#endif //SIMULATIONCONFIG_H
