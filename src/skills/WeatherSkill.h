#pragma once
#include "Skill.h"
#include <windows.h>
#include <winhttp.h>
#include <nlohmann/json.hpp>
#include <string>

#pragma comment(lib, "winhttp.lib")

using json = nlohmann::json;

class WeatherSkill : public Skill {
public:
    std::string getName() const override { return "weather"; }

    std::string getDescription() const override {
        return "获取天气信息。参数: city (城市名称)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"city", "城市名称", "string", true}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "internet"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::NORMAL; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("city");
        if (it == params.end()) {
            return "错误: 缺少 city 参数";
        }

        return getWeather(it->second);
    }

private:
    std::string getWeather(const std::string& city) {
        // 使用 wttr.in API（免费，无需 API Key）
        std::string url = "https://wttr.in/" + urlEncode(city) + "?format=j1";

        std::string response = httpGet(url);
        if (response.empty()) {
            return "无法获取天气信息，请检查网络连接";
        }

        try {
            json data = json::parse(response);
            if (data.contains("current_condition") && !data["current_condition"].empty()) {
                auto current = data["current_condition"][0];
                std::string temp = current.value("temp_C", "?");
                std::string feelsLike = current.value("FeelsLikeC", "?");
                std::string humidity = current.value("humidity", "?");
                std::string weatherDesc = current["weatherDesc"][0].value("value", "?");

                std::stringstream ss;
                ss << "🌤️ " << city << " 天气:\n";
                ss << "  天气: " << weatherDesc << "\n";
                ss << "  温度: " << temp << "°C\n";
                ss << "  体感: " << feelsLike << "°C\n";
                ss << "  湿度: " << humidity << "%\n";

                return ss.str();
            }
        }
        catch (const std::exception& e) {
            return "解析天气数据失败: " + std::string(e.what());
        }

        return "未找到 " + city + " 的天气信息";
    }

    std::string urlEncode(const std::string& str) {
        std::string encoded;
        for (char c : str) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded += c;
            }
            else if (c == ' ') {
                encoded += '+';
            }
            else {
                char buf[4];
                sprintf_s(buf, "%%%02X", (unsigned char)c);
                encoded += buf;
            }
        }
        return encoded;
    }

    std::string httpGet(const std::string& url) {
        // 解析 URL
        std::string host = "wttr.in";
        std::string path = "/" + url.substr(url.find('/') + 1);

        HINTERNET hSession = WinHttpOpen(L"CloseCrab/1.0",
            WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME,
            WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return "";

        HINTERNET hConnect = WinHttpConnect(hSession, std::wstring(host.begin(), host.end()).c_str(), 443, 0);
        if (!hConnect) {
            WinHttpCloseHandle(hSession);
            return "";
        }

        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET",
            std::wstring(path.begin(), path.end()).c_str(),
            NULL, NULL, NULL, WINHTTP_FLAG_SECURE);
        if (!hRequest) {
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return "";
        }

        WinHttpSendRequest(hRequest, NULL, 0, NULL, 0, 0, 0);
        WinHttpReceiveResponse(hRequest, NULL);

        std::string response;
        DWORD dwSize = 0;
        do {
            dwSize = 0;
            if (!WinHttpQueryDataAvailable(hRequest, &dwSize)) break;
            if (dwSize == 0) break;

            std::vector<char> buffer(dwSize + 1);
            DWORD dwRead = 0;
            if (WinHttpReadData(hRequest, buffer.data(), dwSize, &dwRead)) {
                response.append(buffer.data(), dwRead);
            }
        } while (dwSize > 0);

        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);

        return response;
    }
};