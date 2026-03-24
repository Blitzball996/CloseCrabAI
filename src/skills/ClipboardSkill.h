#pragma once
#include "Skill.h"
#include <windows.h>
#include <string>

class ClipboardSkill : public Skill {
public:
    std::string getName() const override { return "clipboard"; }

    std::string getDescription() const override {
        return "操作剪贴板。参数: action (get/set), content (写入内容)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"action", "操作类型 (get/set)", "string", true},
            {"content", "要复制的内容 (仅set时需要)", "string", false}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "system"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::SAFE; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("action");
        if (it == params.end()) {
            return "错误: 缺少 action 参数 (get/set)";
        }

        std::string action = it->second;

        if (action == "get") {
            return getClipboard();
        }
        else if (action == "set") {
            auto contentIt = params.find("content");
            if (contentIt == params.end()) {
                return "错误: 缺少 content 参数";
            }
            return setClipboard(contentIt->second);
        }
        else {
            return "错误: action 必须是 get 或 set";
        }
    }

private:
    std::string getClipboard() {
        if (!OpenClipboard(nullptr)) {
            return "无法打开剪贴板";
        }

        HANDLE hData = GetClipboardData(CF_TEXT);
        if (hData == nullptr) {
            CloseClipboard();
            return "剪贴板为空或无文本数据";
        }

        char* pszText = (char*)GlobalLock(hData);
        if (pszText == nullptr) {
            CloseClipboard();
            return "无法读取剪贴板数据";
        }

        std::string result(pszText);
        GlobalUnlock(hData);
        CloseClipboard();

        return result;
    }

    std::string setClipboard(const std::string& content) {
        if (!OpenClipboard(nullptr)) {
            return "无法打开剪贴板";
        }

        EmptyClipboard();

        // 分配内存
        HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, content.size() + 1);
        if (hGlobal == nullptr) {
            CloseClipboard();
            return "内存分配失败";
        }

        char* pszText = (char*)GlobalLock(hGlobal);
        memcpy(pszText, content.c_str(), content.size());
        pszText[content.size()] = '\0';
        GlobalUnlock(hGlobal);

        SetClipboardData(CF_TEXT, hGlobal);
        CloseClipboard();

        return "已复制到剪贴板: " + content.substr(0, 100) + (content.size() > 100 ? "..." : "");
    }
};