#pragma once
#include "Skill.h"
#include <cstdlib>
#include <array>
#include <memory>
#include <numeric>

class ExecuteCommandSkill : public Skill {
public:
    std::string getName() const override { return "execute_command"; }

    std::string getDescription() const override {
        return "执行系统命令。参数: command (要执行的命令)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"command", "要执行的命令", "string", true}
        };
    }

    bool needsConfirmation() const override { return true; }
    std::string getCategory() const override { return "system"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::UNSAFE; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("command");
        if (it == params.end()) {
            return "错误: 缺少 command 参数";
        }

        std::string cmd = it->second;

        // 安全限制：只允许安全的命令
        std::vector<std::string> allowedCommands = {
            "dir", "ls", "echo", "whoami", "date", "time", "type", "del", "copy", "mkdir", "rmdir"
        };

        bool allowed = false;
        for (const auto& allowedCmd : allowedCommands) {
            if (cmd.find(allowedCmd) == 0) {
                allowed = true;
                break;
            }
        }

        if (!allowed) {
            std::string allowedList;
            for (const auto& ac : allowedCommands) {
                allowedList += ac + ", ";
            }
            return "安全限制: 只允许执行以下命令: " + allowedList;
        }

        // 使用 cmd.exe /c 执行命令，支持重定向
        std::string fullCmd = "cmd.exe /c " + cmd;

        std::array<char, 128> buffer;
        std::string result;

        FILE* pipe = _popen(fullCmd.c_str(), "r");
        if (!pipe) {
            return "无法执行命令: " + cmd;
        }

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }

        _pclose(pipe);

        if (result.empty()) {
            return "命令执行完成";
        }

        if (result.length() > 2000) {
            result = result.substr(0, 2000) + "\n... (输出已截断)";
        }

        return result;
    }
};