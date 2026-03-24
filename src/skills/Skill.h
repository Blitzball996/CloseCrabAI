#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>

// 权限级别（只定义一次）
enum class PermissionLevel {
    SAFE = 0,       // 安全操作（读文件、只读命令）
    NORMAL = 1,     // 普通操作（写文件、运行程序）
    DANGEROUS = 2,  // 危险操作（系统命令、删除文件）
    UNSAFE = 3      // 不安全操作（需要确认）
};

struct SkillParameter {
    std::string name;
    std::string description;
    std::string type;
    bool required;
};

class Skill {
public:
    virtual ~Skill() = default;

    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;
    virtual std::vector<SkillParameter> getParameters() const = 0;
    virtual std::string execute(const std::map<std::string, std::string>& params) = 0;
    virtual bool needsConfirmation() const { return false; }
    virtual std::string getCategory() const { return "general"; }
    virtual PermissionLevel getPermissionLevel() const {
        return PermissionLevel::NORMAL;
    }

    // 安全执行（带沙箱）
    std::string safeExecute(const std::map<std::string, std::string>& params);
};