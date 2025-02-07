//
// Created by ace on 2/7/25.
//

#ifndef IMGUIWIDGETS_H
#define IMGUIWIDGETS_H

#include <functional>
#include <vector>
#include "imgui.h"


class ImGuiWidgets
{
public:
    template<typename CONFIG_TYPE>
    static void RenderComboBox(
        std::vector<std::pair<char*, CONFIG_TYPE>> items,
        const std::function<void(CONFIG_TYPE)>& config_setter,
        const std::function<void(char*)>& ui_config_setter,
        const char* combo_label,
        const char* combo_preview_value
        )
    {
        if(ImGui::BeginCombo(combo_label,combo_preview_value))
        {

            for (auto& [name,value] : items)
            {
                if (ImGui::Selectable(name,false))
                {
                    config_setter(value);
                    ui_config_setter(name);
                }
            }
            ImGui::EndCombo();
        }
    }
};


#endif //IMGUIWIDGETS_H
