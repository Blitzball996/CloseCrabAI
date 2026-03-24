#pragma once
#include "Skill.h"
#include <fstream>
#include <sstream>

class ReadFileSkill : public Skill {
public:
    std::string getName() const override { return "read_file"; }

    std::string getDescription() const override {
        return "读取文件内容。参数: file_path (文件路径)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"file_path", "要读取的文件路径", "string", true}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "file"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::SAFE; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("file_path");
        if (it == params.end()) {
            return "错误: 缺少 file_path 参数";
        }

        std::string filePath = it->second;
        std::ifstream file(filePath);

        if (!file.is_open()) {
            return "无法打开文件: " + filePath;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        if (content.empty()) {
            return "文件为空";
        }

        if (content.length() > 2000) {
            content = content.substr(0, 2000) + "\n... (内容已截断)";
        }

        return content;
    }
};