#pragma once
#include "Skill.h"
#include <fstream>

class WriteFileSkill : public Skill {
public:
    std::string getName() const override { return "write_file"; }

    std::string getDescription() const override {
        return "写入内容到文件。参数: file_path (文件路径), content (要写入的内容)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"file_path", "文件路径", "string", true},
            {"content", "要写入的内容", "string", true}
        };
    }

    bool needsConfirmation() const override { return true; }
    std::string getCategory() const override { return "file"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::DANGEROUS; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto itPath = params.find("file_path");
        auto itContent = params.find("content");

        if (itPath == params.end()) {
            return "错误: 缺少 file_path 参数";
        }
        if (itContent == params.end()) {
            return "错误: 缺少 content 参数";
        }

        std::ofstream file(itPath->second);
        if (!file.is_open()) {
            return "无法写入文件: " + itPath->second;
        }

        file << itContent->second;
        file.close();

        return "成功写入文件: " + itPath->second;
    }
};