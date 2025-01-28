//
// Created by ace on 2025.01.15..
//

#ifndef SIMULATIONCONFIG_H
#define SIMULATIONCONFIG_H
#include <utility>

#include "BarnesHutConfig.h"

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

struct SimulationConfig {

public:
    SimulationConfig() = default;

    SimulationConfig(const SimulationConfig& other)
        : num_of_bodies(other.num_of_bodies),
          gravitational_constant(other.gravitational_constant),
          starting_speed_mul(other.starting_speed_mul),
          starting_volume_radius(other.starting_volume_radius),
          number_of_massive_objects(other.number_of_massive_objects),
          massive_object_mass(other.massive_object_mass),
          pos_config(other.pos_config),
          vel_config(other.vel_config),
          mass_distr(other.mass_distr),
          algo_config(other.algo_config),
          barnes_hut_config(other.barnes_hut_config)
    {
    }

    SimulationConfig(SimulationConfig&& other) noexcept
        : num_of_bodies(other.num_of_bodies),
          gravitational_constant(other.gravitational_constant),
          starting_speed_mul(other.starting_speed_mul),
          starting_volume_radius(other.starting_volume_radius),
          number_of_massive_objects(other.number_of_massive_objects),
          massive_object_mass(other.massive_object_mass),
          pos_config(other.pos_config),
          vel_config(other.vel_config),
          mass_distr(other.mass_distr),
          algo_config(other.algo_config),
          barnes_hut_config(std::move(other.barnes_hut_config))
    {
    }

    SimulationConfig& operator=(const SimulationConfig& other)
    {
        if (this == &other)
            return *this;
        num_of_bodies = other.num_of_bodies;
        gravitational_constant = other.gravitational_constant;
        starting_speed_mul = other.starting_speed_mul;
        starting_volume_radius = other.starting_volume_radius;
        number_of_massive_objects = other.number_of_massive_objects;
        massive_object_mass = other.massive_object_mass;
        pos_config = other.pos_config;
        vel_config = other.vel_config;
        mass_distr = other.mass_distr;
        algo_config = other.algo_config;
        barnes_hut_config = other.barnes_hut_config;
        return *this;
    }

    SimulationConfig& operator=(SimulationConfig&& other) noexcept
    {
        if (this == &other)
            return *this;
        num_of_bodies = other.num_of_bodies;
        gravitational_constant = other.gravitational_constant;
        starting_speed_mul = other.starting_speed_mul;
        starting_volume_radius = other.starting_volume_radius;
        number_of_massive_objects = other.number_of_massive_objects;
        massive_object_mass = other.massive_object_mass;
        pos_config = other.pos_config;
        vel_config = other.vel_config;
        mass_distr = std::move(other.mass_distr);
        algo_config = other.algo_config;
        barnes_hut_config = other.barnes_hut_config;
        return *this;
    }

    [[nodiscard]] int& GetNumberOfBodies()
    {
        return num_of_bodies;
    }

    [[nodiscard]] float& GetGravitationalConstant()
    {
        return gravitational_constant;
    }

    [[nodiscard]] float& GetStartingSpeedMul()
    {
        return starting_speed_mul;
    }

    [[nodiscard]] float& GetStartingVolumeRadius()
    {
        return starting_volume_radius;
    }

    [[nodiscard]] int& GetNumberOfMassiveObjects()
    {
        return number_of_massive_objects;
    }

    [[nodiscard]] float& GetMassiveObjectMass()
    {
        return massive_object_mass;
    }

    [[nodiscard]] PositionConfig& GetPositionConfig()
    {
        return pos_config;
    }

    [[nodiscard]] VelocityConfig& GetVelocityConfig()
    {
        return vel_config;
    }

    [[nodiscard]] NormalDistribution& GetMassDistribution()
    {
        return mass_distr;
    }

    [[nodiscard]] AlgorithmConfig& GetAlgorithmConfig(){
        return algo_config;
    }

    [[nodiscard]] BarnesHutConfig& GetBarnesHutConfig(){
        return barnes_hut_config;
    }

    void SetNumberOfBodies(const int number_of_particles)
    {
        this->num_of_bodies = number_of_particles;
    }

    void SetGravitationalConstant(const float gravitational_constant)
    {
        this->gravitational_constant = gravitational_constant;
    }

    void SetStartingSpeedMul(const float starting_speed_mul)
    {
        this->starting_speed_mul = starting_speed_mul;
    }

    void SetStartingVolumeRadius(const float starting_volume_radius)
    {
        this->starting_volume_radius = starting_volume_radius;
    }

    void SetNumberOfMassiveObjects(const int number_of_massive_objects)
    {
        this->number_of_massive_objects = number_of_massive_objects;
    }

    void SetMassiveObjectMass(const float massive_object_mass)
    {
        this->massive_object_mass = massive_object_mass;
    }

    void SetPositionConfig(const PositionConfig pos_config)
    {
        this->pos_config = pos_config;
    }

    void SetVelocityConfig(const VelocityConfig vel_config)
    {
        this->vel_config = vel_config;
    }

    void SetMassDistribution(const NormalDistribution& mass_distr)
    {
        this->mass_distr = mass_distr;
    }

    void SetAlgorithmConfig(const AlgorithmConfig algo)
    {
        algo_config = algo;
    }

    void SetBarnesHutConfig(const BarnesHutConfig barnes_hut_config)
    {
        this->barnes_hut_config = barnes_hut_config;
    }

private:
    int num_of_bodies = 20000;
    float gravitational_constant = 6.67e-11;

    float starting_speed_mul = 1.0f;
    float starting_volume_radius = 1.5f;

    int number_of_massive_objects = 0;
    float massive_object_mass = 1.0f;

    PositionConfig pos_config = UNIFORM_POS;
    VelocityConfig vel_config = FUNC_ZERO_VEL;
    AlgorithmConfig algo_config = BRUTE_FORCE_GLOBAL;

    NormalDistribution mass_distr = NormalDistribution(0.5,0.25);

    BarnesHutConfig barnes_hut_config;
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
