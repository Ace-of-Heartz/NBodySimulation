//
// Created by ace on 2025.01.15..
//

#ifndef SIMULATIONCONFIG_H
#define SIMULATIONCONFIG_H
#include <utility>

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
    FUNC_ZERO_VEL
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
        : number_of_particles(other.number_of_particles),
          gravitational_constant(other.gravitational_constant),
          starting_speed_mul(other.starting_speed_mul),
          starting_volume_radius(other.starting_volume_radius),
          number_of_massive_objects(other.number_of_massive_objects),
          massive_object_mass(other.massive_object_mass),
          pos_config(other.pos_config),
          vel_config(other.vel_config),
          mass_distr(other.mass_distr)
    {
    }

    SimulationConfig(SimulationConfig&& other) noexcept
        : number_of_particles(other.number_of_particles),
          gravitational_constant(other.gravitational_constant),
          starting_speed_mul(other.starting_speed_mul),
          starting_volume_radius(other.starting_volume_radius),
          number_of_massive_objects(other.number_of_massive_objects),
          massive_object_mass(other.massive_object_mass),
          pos_config(other.pos_config),
          vel_config(other.vel_config),
          mass_distr(std::move(other.mass_distr))
    {
    }

    SimulationConfig& operator=(const SimulationConfig& other)
    {
        if (this == &other)
            return *this;
        number_of_particles = other.number_of_particles;
        gravitational_constant = other.gravitational_constant;
        starting_speed_mul = other.starting_speed_mul;
        starting_volume_radius = other.starting_volume_radius;
        number_of_massive_objects = other.number_of_massive_objects;
        massive_object_mass = other.massive_object_mass;
        pos_config = other.pos_config;
        vel_config = other.vel_config;
        mass_distr = other.mass_distr;
        return *this;
    }

    SimulationConfig& operator=(SimulationConfig&& other) noexcept
    {
        if (this == &other)
            return *this;
        number_of_particles = other.number_of_particles;
        gravitational_constant = other.gravitational_constant;
        starting_speed_mul = other.starting_speed_mul;
        starting_volume_radius = other.starting_volume_radius;
        number_of_massive_objects = other.number_of_massive_objects;
        massive_object_mass = other.massive_object_mass;
        pos_config = other.pos_config;
        vel_config = other.vel_config;
        mass_distr = std::move(other.mass_distr);
        return *this;
    }

    [[nodiscard]] int& GetNumberOfParticles()
    {
        return number_of_particles;
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

    void SetNumberOfParticles(const int number_of_particles)
    {
        this->number_of_particles = number_of_particles;
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

private:
    int number_of_particles = 20000;
    float gravitational_constant = 6.67e-11;

    float starting_speed_mul = 1.0f;
    float starting_volume_radius = 0.5f;

    int number_of_massive_objects = 0;
    float massive_object_mass = 1.0f;

    PositionConfig pos_config = UNIFORM_POS;
    VelocityConfig vel_config = FUNC_ZERO_VEL;

    NormalDistribution mass_distr = NormalDistribution(0.5,0.25);
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
