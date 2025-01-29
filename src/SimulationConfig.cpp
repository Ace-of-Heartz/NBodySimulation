//
// Created by ace on 2025.01.15..
//

#include "SimulationConfig.h"

#include <algorithm>


SimulationConfig::SimulationConfig(const SimulationConfig& other)
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
      barnes_hut_config(other.barnes_hut_config),
{
}

SimulationConfig::SimulationConfig(SimulationConfig&& other) noexcept
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
      barnes_hut_config(std::move(other.barnes_hut_config)),
{
}

SimulationConfig& SimulationConfig::operator=(const SimulationConfig& other)
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

SimulationConfig& SimulationConfig::operator=(SimulationConfig&& other) noexcept
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

[[nodiscard]] int& SimulationConfig::GetNumberOfBodies()
{
    return num_of_bodies;
}

[[nodiscard]] float& SimulationConfig::GetGravitationalConstant()
{
    return gravitational_constant;
}

[[nodiscard]] float& SimulationConfig::GetStartingSpeedMul()
{
    return starting_speed_mul;
}

[[nodiscard]] float& SimulationConfig::GetStartingVolumeRadius()
{
    return starting_volume_radius;
}

[[nodiscard]] int& SimulationConfig::GetNumberOfMassiveObjects()
{
    return number_of_massive_objects;
}

[[nodiscard]] float& SimulationConfig::GetMassiveObjectMass()
{
    return massive_object_mass;
}

[[nodiscard]] PositionConfig& SimulationConfig::GetPositionConfig()
{
    return pos_config;
}

[[nodiscard]] VelocityConfig& SimulationConfig::GetVelocityConfig()
{
    return vel_config;
}

[[nodiscard]] NormalDistribution& SimulationConfig::GetMassDistribution()
{
    return mass_distr;
}

[[nodiscard]] AlgorithmConfig& SimulationConfig::GetAlgorithmConfig()
{
    return algo_config;
}

[[nodiscard]] BarnesHutConfig& SimulationConfig::GetBarnesHutConfig()
{
    return barnes_hut_config;
}

void SimulationConfig::SetNumberOfBodies(const int number_of_particles)
{
    this->num_of_bodies = number_of_particles;
}

void SimulationConfig::SetGravitationalConstant(const float gravitational_constant)
{
    this->gravitational_constant = gravitational_constant;
}

void SimulationConfig::SetStartingSpeedMul(const float starting_speed_mul)
{
    this->starting_speed_mul = starting_speed_mul;
}

void SimulationConfig::SetStartingVolumeRadius(const float starting_volume_radius)
{
    this->starting_volume_radius = starting_volume_radius;
}

void SimulationConfig::SetNumberOfMassiveObjects(const int number_of_massive_objects)
{
    this->number_of_massive_objects = number_of_massive_objects;
}

void SimulationConfig::SetMassiveObjectMass(const float massive_object_mass)
{
    this->massive_object_mass = massive_object_mass;
}

void SimulationConfig::SetPositionConfig(const PositionConfig pos_config)
{
    this->pos_config = pos_config;
}

void SimulationConfig::SetVelocityConfig(const VelocityConfig vel_config)
{
    this->vel_config = vel_config;
}

void SimulationConfig::SetMassDistribution(const NormalDistribution& mass_distr)
{
    this->mass_distr = mass_distr;
}

void SimulationConfig::SetAlgorithmConfig(const AlgorithmConfig algo)
{
    algo_config = algo;
}

void SimulationConfig::SetBarnesHutConfig(const BarnesHutConfig barnes_hut_config)
{
    this->barnes_hut_config = barnes_hut_config;
}
