#include "Skill.h"
#include "security/Sandbox.h"

std::string Skill::safeExecute(const std::map<std::string, std::string>& params) {
    auto& sandbox = Sandbox::getInstance();

    // 构建动作描述
    std::string action = getName();
    for (const auto& p : params) {
        action += " " + p.first + "=" + p.second;
    }
    if (action.length() > 100) {
        action = action.substr(0, 100) + "...";
    }

    return sandbox.executeSkill(
        getName(),
        action,
        getPermissionLevel(),
        [this, &params]() { return this->execute(params); }
    );
}