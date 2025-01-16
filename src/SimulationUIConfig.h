//
// Created by ace on 2025.01.15..
//

#ifndef SIMULATIONUICONFIG_H
#define SIMULATIONUICONFIG_H
#include <utility>
#include <vector>

#include "SimulationConfig.h"


struct SimulationUIConfig {
public:
    SimulationUIConfig() = default;

    SimulationUIConfig(const SimulationUIConfig& other)
        : position_distr_item(other.position_distr_item),
          velocity_distr_item(other.velocity_distr_item)
    {
    }

    SimulationUIConfig(SimulationUIConfig&& other) noexcept
        : position_distr_item(other.position_distr_item),
          velocity_distr_item(other.velocity_distr_item)
    {
    }

    SimulationUIConfig& operator=(const SimulationUIConfig& other)
    {
        if (this == &other)
            return *this;
        position_distr_item = other.position_distr_item;
        velocity_distr_item = other.velocity_distr_item;
        return *this;
    }

    SimulationUIConfig& operator=(SimulationUIConfig&& other) noexcept
    {
        if (this == &other)
            return *this;
        position_distr_item = other.position_distr_item;
        velocity_distr_item = other.velocity_distr_item;
        return *this;
    }

    [[nodiscard]] const char* GetPositionConfigItem() const
    {
        return position_distr_item;
    }

    [[nodiscard]] const char* GetVelocityConfigItem() const
    {
        return velocity_distr_item;
    }

    void SetPositionConfigItem(const char* const pos_config_item)
    {
        this->position_distr_item = pos_config_item;
    }

    void SetVelocityConfigItem(const char* const vel_config_item)
    {
        this->velocity_distr_item = vel_config_item;
    }

private:
    const char* position_distr_item = "Uniform Distribution";
    const char* velocity_distr_item = "Functionally Zero";
};

struct SimulationUI
{
    SimulationUI() = default;

    [[nodiscard]] SimulationUIConfig GetUIConfig() const
    {
        return config;
    }

    void SetUIConfig(SimulationUIConfig config) {config = config;}


    static inline const std::vector<std::pair<char*, PositionConfig>> pos_config_items = {{"Distribute in Sphere",SPHERE_POS}, {"Uniform Distribution",UNIFORM_POS}};
    static inline const std::vector<std::pair<char*,VelocityConfig>> vel_config_items = {{"Random",RANDOM_VEL},{"Starting outwards",STARTING_OUT_VEL},{"Starting inwards",STARTING_IN_VEL},{"Functionally Zero",FUNC_ZERO_VEL}};

private:
    SimulationUIConfig config;
};


#endif //SIMULATIONUICONFIG_H
