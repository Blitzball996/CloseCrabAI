#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

// 前向声明 PermissionLevel（定义在 Skill.h 中）
enum class PermissionLevel;

// 权限请求回调
using PermissionCallback = std::function<bool(const std::string& skill,
    const std::string& action,
    PermissionLevel level)>;

class Sandbox {
public:
    static Sandbox& getInstance();

    // 设置安全模式
    enum class Mode {
        DISABLED = 0,   // 完全禁用沙箱（直接执行）
        ASK = 1,        // 每次询问用户
        AUTO = 2,       // 自动拒绝危险操作
        TRUSTED = 3     // 信任模式（只记录，不拦截）
    };

    void setMode(Mode mode);
    Mode getMode() const;
    void setPermissionCallback(PermissionCallback callback);

    std::string executeSkill(const std::string& skillName,
        const std::string& action,
        PermissionLevel level,
        std::function<std::string()> executor);

    void addWhitelist(const std::string& skill, const std::string& action);
    void addBlacklist(const std::string& skill, const std::string& action);
    std::vector<std::string> getAuditLog() const;
    void clearAuditLog();

private:
    Sandbox() = default;
    Mode currentMode = Mode::ASK;
    PermissionCallback permissionCallback;
    std::vector<std::pair<std::string, std::string>> whitelist;
    std::vector<std::pair<std::string, std::string>> blacklist;
    std::vector<std::string> auditLog;

    void log(const std::string& entry);
    bool checkPermission(const std::string& skill,
        const std::string& action,
        PermissionLevel level);
    bool isWhitelisted(const std::string& skill, const std::string& action) const;
    bool isBlacklisted(const std::string& skill, const std::string& action) const;
};