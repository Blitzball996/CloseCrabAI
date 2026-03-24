#pragma once
#include "Skill.h"
#include <cstdlib>
#include <string>
#include <map>

class OpenAppSkill : public Skill {
public:
    std::string getName() const override { return "open_app"; }

    std::string getDescription() const override {
        return "打开指定的应用程序或网页。参数: app_name (程序名: notepad, calc, edge, chrome, 百度, 谷歌), url (可选，要打开的网址)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"app_name", "要打开的程序名称或浏览器名", "string", true},
            {"url", "要打开的网址（可选）", "string", false}
        };
    }

    bool needsConfirmation() const override { return true; }
    std::string getCategory() const override { return "system"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::NORMAL; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("app_name");
        if (it == params.end()) {
            return "错误: 缺少 app_name 参数";
        }

        std::string appName = it->second;
        std::string url;
        auto urlIt = params.find("url");
        if (urlIt != params.end()) {
            url = urlIt->second;
        }

        // 如果没有指定 URL，检查是否是中文搜索引擎名
        if (url.empty()) {
            if (appName == "百度" || appName == "baidu") {
                url = "https://www.baidu.com";
            }
            else if (appName == "谷歌" || appName == "google") {
                url = "https://www.google.com";
            }
            else if (appName == "必应" || appName == "bing") {
                url = "https://www.bing.com";
            }
        }

        // 浏览器名称映射
        std::string browserName = appName;
        if (appName == "百度" || appName == "baidu") browserName = "edge";
        else if (appName == "谷歌" || appName == "google") browserName = "edge";
        else if (appName == "必应" || appName == "bing") browserName = "edge";

        std::string command;

        // 浏览器映射
        if (browserName == "edge" || browserName == "microsoft edge" || browserName == "Edge" || browserName == "Microsoft Edge") {
            if (!url.empty()) {
                command = "start microsoft-edge:" + url;
            }
            else {
                command = "start microsoft-edge:";
            }
        }
        else if (browserName == "chrome" || browserName == "google chrome" || browserName == "Google Chrome") {
            if (!url.empty()) {
                command = "start chrome \"" + url + "\"";
            }
            else {
                command = "start chrome";
            }
        }
        else if (browserName == "firefox" || browserName == "Firefox") {
            if (!url.empty()) {
                command = "start firefox \"" + url + "\"";
            }
            else {
                command = "start firefox";
            }
        }
        else if (appName == "notepad" || appName == "记事本") {
            command = "notepad.exe";
        }
        else if (appName == "calc" || appName == "计算器") {
            command = "calc.exe";
        }
        else if (appName == "explorer" || appName == "文件管理器") {
            command = "explorer.exe";
        }
        else {
            if (!url.empty()) {
                command = "start " + appName + " \"" + url + "\"";
            }
            else {
                command = "start " + appName;
            }
        }

        int result = system(command.c_str());
        if (result == 0) {
            std::string msg = "已打开 " + appName;
            if (!url.empty()) {
                msg += "，正在访问 " + url;
            }
            return msg;
        }
        else {
            return "无法打开 " + appName;
        }
    }
};