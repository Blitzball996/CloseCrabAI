#pragma once
#include "Skill.h"
#include <string>
#include <map>
#include <vector>

class WebSearchSkill : public Skill {
public:
    std::string getName() const override { return "search_web"; }

    std::string getDescription() const override {
        return "搜索网络获取信息。参数: query (搜索关键词)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"query", "要搜索的关键词", "string", true}
        };
    }

    bool needsConfirmation() const override { return true; }
    std::string getCategory() const override { return "internet"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::NORMAL; }

    std::string execute(const std::map<std::string, std::string>& params) override;

private:
    std::string searchWeb(const std::string& query);
};