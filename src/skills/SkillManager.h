#pragma once
#include "Skill.h"
#include <memory>
#include <map>
#include <vector>

// Skill 执行模式
enum class SkillMode {
    AUTO = 0,  // 只聊天，不调用 Skill 
    CHAT_ONLY = 1,       // AI 自动判断
    SKILL_ONLY = 2, // 只执行 Skill，不聊天
    ASK = 3         // 每次询问
};

class SkillManager {
public:
    static SkillManager& getInstance();

    // 注册技能
    void registerSkill(std::unique_ptr<Skill> skill);

    // 获取所有技能描述（给 AI 看）
    std::string getSkillsDescription() const;

    // 执行技能
    std::string executeSkill(const std::string& skillName,
        const std::map<std::string, std::string>& params);

    // 获取技能
    Skill* getSkill(const std::string& name);

    // 列出所有技能
    std::vector<std::string> listSkills() const;

    // 新增：模式控制
    void setMode(SkillMode mode);
    SkillMode getMode() const;
    std::string getModeName() const;

    // 新增：判断是否应该执行 Skill
    bool shouldExecuteSkill(const std::string& aiResponse, std::string& skillName,
        std::map<std::string, std::string>& params);

    // 新增：解析 AI 回复中的 Skill 调用
    bool parseSkillCall(const std::string& response, std::string& skillName,
        std::map<std::string, std::string>& params);

private:
    SkillManager() = default;
    std::map<std::string, std::unique_ptr<Skill>> skills;
    SkillMode currentMode = SkillMode::CHAT_ONLY;
};