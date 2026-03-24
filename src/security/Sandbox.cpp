#include "Sandbox.h"
#include "skills/Skill.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

Sandbox& Sandbox::getInstance() {
    static Sandbox instance;
    return instance;
}

void Sandbox::setMode(Mode mode) {
    currentMode = mode;
    std::string modeStr;
    switch (mode) {
    case Mode::DISABLED: modeStr = "DISABLED"; break;
    case Mode::ASK: modeStr = "ASK"; break;
    case Mode::AUTO: modeStr = "AUTO"; break;
    case Mode::TRUSTED: modeStr = "TRUSTED"; break;
    }
    spdlog::info("Sandbox mode set to: {}", modeStr);
}

Sandbox::Mode Sandbox::getMode() const {
    return currentMode;
}

void Sandbox::setPermissionCallback(PermissionCallback callback) {
    permissionCallback = callback;
}

std::string Sandbox::executeSkill(const std::string& skillName,
    const std::string& action,
    PermissionLevel level,
    std::function<std::string()> executor) {
    // 记录尝试
    std::string logEntry = "Skill: " + skillName + ", Action: " + action;
    log(logEntry);

    // 检查权限
    if (!checkPermission(skillName, action, level)) {
        std::string msg = "权限不足: " + skillName + " - " + action;
        log(msg);
        return "[安全沙箱] " + msg;
    }

    // 执行
    try {
        std::string result = executor();
        log("执行成功: " + skillName + " -> " + result.substr(0, 100));
        return result;
    }
    catch (const std::exception& e) {
        log("执行失败: " + skillName + " - " + e.what());
        return "[安全沙箱] 执行失败: " + std::string(e.what());
    }
}

bool Sandbox::checkPermission(const std::string& skill,
    const std::string& action,
    PermissionLevel level) {
    // 检查黑名单
    if (isBlacklisted(skill, action)) {
        log("黑名单拦截: " + skill + " - " + action);
        return false;
    }

    // 检查白名单
    if (isWhitelisted(skill, action)) {
        log("白名单通过: " + skill + " - " + action);
        return true;
    }

    // 根据模式判断
    switch (currentMode) {
    case Mode::DISABLED:
        log("沙箱禁用，直接执行: " + skill + " - " + action);
        return true;

    case Mode::TRUSTED:
        log("信任模式，允许执行: " + skill + " - " + action);
        return true;

    case Mode::AUTO:
        if (level == PermissionLevel::SAFE) {
            return true;
        }
        else if (level == PermissionLevel::NORMAL) {
            log("自动允许普通操作: " + skill + " - " + action);
            return true;
        }
        else {
            log("自动拒绝危险操作: " + skill + " - " + action);
            return false;
        }

    case Mode::ASK:
        if (permissionCallback) {
            return permissionCallback(skill, action, level);
        }
        else {
            // 没有回调时，危险操作默认拒绝
            if (level >= PermissionLevel::DANGEROUS) {
                std::cout << "\n[安全沙箱] " << skill << " 想要执行: " << action;
                std::cout << "\n是否允许? (y/n): ";
                std::string answer;
                std::getline(std::cin, answer);
                return (answer == "y" || answer == "Y");
            }
            return true;
        }

    default:
        return false;
    }
}

void Sandbox::addWhitelist(const std::string& skill, const std::string& action) {
    whitelist.emplace_back(skill, action);
    spdlog::info("Added to whitelist: {} - {}", skill, action);
}

void Sandbox::addBlacklist(const std::string& skill, const std::string& action) {
    blacklist.emplace_back(skill, action);
    spdlog::info("Added to blacklist: {} - {}", skill, action);
}

std::vector<std::string> Sandbox::getAuditLog() const {
    return auditLog;
}

void Sandbox::clearAuditLog() {
    auditLog.clear();
    spdlog::info("Audit log cleared");
}

void Sandbox::log(const std::string& entry) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << " | " << entry;
    auditLog.push_back(ss.str());
    spdlog::debug("Sandbox: {}", entry);
}

bool Sandbox::isWhitelisted(const std::string& skill, const std::string& action) const {
    for (const auto& item : whitelist) {
        if (item.first == skill && (item.second == "*" || item.second == action)) {
            return true;
        }
    }
    return false;
}

bool Sandbox::isBlacklisted(const std::string& skill, const std::string& action) const {
    for (const auto& item : blacklist) {
        if (item.first == skill && (item.second == "*" || item.second == action)) {
            return true;
        }
    }
    return false;
}