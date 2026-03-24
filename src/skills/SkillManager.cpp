#include "SkillManager.h"
#include "OpenAppSkill.h"
#include "ReadFileSkill.h"
#include "WriteFileSkill.h"
#include "ExecuteCommandSkill.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <regex>
#include <iostream>

SkillManager& SkillManager::getInstance() {
    static SkillManager instance;
    return instance;
}

void SkillManager::registerSkill(std::unique_ptr<Skill> skill) {
    std::string name = skill->getName();
    skills[name] = std::move(skill);
    spdlog::info("Registered skill: {}", name);
}

std::string SkillManager::getSkillsDescription() const {
    std::stringstream ss;
    ss << "Available skills:\n";
    for (const auto& [name, skill] : skills) {
        ss << "  - " << name << ": " << skill->getDescription() << "\n";
        ss << "    Parameters: ";
        for (const auto& param : skill->getParameters()) {
            ss << param.name << " (" << param.type << ")" << (param.required ? " [required]" : " [optional]") << ", ";
        }
        ss << "\n";
    }
    return ss.str();
}

std::string SkillManager::executeSkill(const std::string& skillName,
    const std::map<std::string, std::string>& params) {
    auto it = skills.find(skillName);
    if (it == skills.end()) {
        spdlog::warn("Skill not found: {}", skillName);
        return "错误: 未找到技能 '" + skillName + "'";
    }

    spdlog::info("Executing skill: {}", skillName);

    // 使用安全执行
    return it->second->safeExecute(params);
}

Skill* SkillManager::getSkill(const std::string& name) {
    auto it = skills.find(name);
    if (it != skills.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::vector<std::string> SkillManager::listSkills() const {
    std::vector<std::string> result;
    for (const auto& [name, _] : skills) {
        result.push_back(name);
    }
    return result;
}

void SkillManager::setMode(SkillMode mode) {
    currentMode = mode;
    spdlog::info("Skill mode set to: {}", getModeName());
}

SkillMode SkillManager::getMode() const {
    return currentMode;
}

std::string SkillManager::getModeName() const {
    switch (currentMode) {
    case SkillMode::AUTO: return "AUTO (AI decides)";
    case SkillMode::CHAT_ONLY: return "CHAT_ONLY (no skills)";
    case SkillMode::SKILL_ONLY: return "SKILL_ONLY (only skills)";
    case SkillMode::ASK: return "ASK (ask before skill)";
    default: return "UNKNOWN";
    }
}

bool SkillManager::parseSkillCall(const std::string& response,
    std::string& skillName,
    std::map<std::string, std::string>& params) {
    // 查找 SKILL: 标记
    size_t skillPos = response.find("SKILL:");
    if (skillPos == std::string::npos) {
        return false;
    }

    // 提取技能部分
    std::string skillPart = response.substr(skillPos);

    // 解析技能名
    size_t nameStart = skillPart.find(':') + 1;
    size_t nameEnd = skillPart.find('\n', nameStart);
    if (nameEnd == std::string::npos) {
        nameEnd = skillPart.length();
    }
    skillName = skillPart.substr(nameStart, nameEnd - nameStart);
    skillName.erase(0, skillName.find_first_not_of(" \t"));
    skillName.erase(skillName.find_last_not_of(" \t") + 1);

    // 查找 PARAMS:
    size_t paramsPos = skillPart.find("PARAMS:");
    if (paramsPos != std::string::npos) {
        std::string paramsStr = skillPart.substr(paramsPos + 7);
        size_t paramsEnd = paramsStr.find('\n');
        if (paramsEnd != std::string::npos) {
            paramsStr = paramsStr.substr(0, paramsEnd);
        }

        // 解析参数 (格式: key=value, key2=value2)
        size_t start = 0;
        while (start < paramsStr.length()) {
            size_t eqPos = paramsStr.find('=', start);
            if (eqPos == std::string::npos) break;

            std::string key = paramsStr.substr(start, eqPos - start);
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);

            size_t commaPos = paramsStr.find(',', eqPos + 1);
            if (commaPos == std::string::npos) {
                commaPos = paramsStr.length();
            }

            std::string value = paramsStr.substr(eqPos + 1, commaPos - eqPos - 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            params[key] = value;
            start = commaPos + 1;
        }
    }

    return true;
}

bool SkillManager::shouldExecuteSkill(const std::string& aiResponse,
    std::string& skillName,
    std::map<std::string, std::string>& params) {
    // 根据模式决定
    switch (currentMode) {
    case SkillMode::CHAT_ONLY:
        spdlog::debug("Skill mode: CHAT_ONLY, ignoring skill call");
        return false;

    case SkillMode::SKILL_ONLY:
        // 只执行技能，不返回聊天内容
        return parseSkillCall(aiResponse, skillName, params);

    case SkillMode::ASK:
        if (parseSkillCall(aiResponse, skillName, params)) {
            // 询问用户
            std::cout << "\n[Skill] AI 想执行: " << skillName << std::endl;
            std::cout << "是否允许? (y/n): ";
            std::string answer;
            std::getline(std::cin, answer);
            return (answer == "y" || answer == "Y");
        }
        return false;

    case SkillMode::AUTO:
    default:
        // 自动模式：如果有 Skill 调用就执行
        return parseSkillCall(aiResponse, skillName, params);
    }
}