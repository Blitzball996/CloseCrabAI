#pragma once
#include "Skill.h"
#include <memory>
#include <functional>

class AnalyzeSkill : public Skill {
public:
    std::string getName() const override { return "analyze"; }

    std::string getDescription() const override {
        return "多轮分析数据，支持链式调用。参数: action (detect/entropy/strings/hex), data, options";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"action", "操作类型: detect/entropy/strings/hex/pattern", "string", true},
            {"data", "数据内容或文件路径", "string", true},
            {"options", "额外选项", "string", false}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "analysis"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::SAFE; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("action");
        auto dataIt = params.find("data");
        if (it == params.end() || dataIt == params.end()) {
            return "错误: 缺少 action 或 data 参数";
        }

        std::string action = it->second;
        std::string data = dataIt->second;
        std::string options = params.count("options") ? params.at("options") : "";

        return analyze(action, data, options);
    }

    // 链式调用支持
    std::string chain(const std::vector<std::pair<std::string, std::string>>& steps) {
        std::string currentData;
        std::string result;

        for (const auto& step : steps) {
            std::string action = step.first;
            std::string params = step.second;

            std::map<std::string, std::string> paramMap;
            paramMap["action"] = action;
            paramMap["data"] = currentData.empty() ? params : currentData;

            result = execute(paramMap);
            currentData = result;
        }

        return result;
    }

private:
    std::string analyze(const std::string& action, const std::string& data, const std::string& options) {
        if (action == "detect") {
            return detectFileType(data);
        }
        else if (action == "entropy") {
            return calculateEntropy(data);
        }
        else if (action == "strings") {
            return extractStrings(data, options);
        }
        else if (action == "hex") {
            return toHex(data);
        }
        else if (action == "pattern") {
            return findPatterns(data);
        }

        return "未知操作: " + action;
    }

    std::string detectFileType(const std::string& data) {
        // 从文件路径读取或从数据检测
        std::vector<unsigned char> bytes;

        // 尝试作为文件路径打开
        std::ifstream file(data, std::ios::binary);
        if (file.is_open()) {
            bytes.resize(256);
            file.read(reinterpret_cast<char*>(bytes.data()), 256);
            file.close();
        }
        else {
            // 作为数据字符串处理
            bytes.assign(data.begin(), data.end());
        }

        return analyzeBinary(bytes);
    }

    std::string analyzeBinary(const std::vector<unsigned char>& data) {
        std::stringstream ss;

        if (data.size() >= 4) {
            if (data[0] == 0x4D && data[1] == 0x5A) {
                ss << "PE (Windows 可执行文件)";
            }
            else if (data[0] == 0x7F && data[1] == 0x45 && data[2] == 0x4C && data[3] == 0x46) {
                ss << "ELF (Linux 可执行文件)";
            }
            else if (data[0] == 0x50 && data[1] == 0x4B) {
                ss << "ZIP 压缩包";
            }
            else if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
                ss << "PNG 图片";
            }
            else {
                ss << "未知类型 (魔数: ";
                for (size_t i = 0; i < std::min((size_t)4, data.size()); ++i) {
                    ss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
                }
                ss << ")";
            }
        }

        return ss.str();
    }

    std::string calculateEntropy(const std::string& data) {
        std::map<char, int> freq;
        for (char c : data) {
            freq[c]++;
        }

        double entropy = 0.0;
        for (const auto& pair : freq) {
            double p = (double)pair.second / data.size();
            entropy -= p * log2(p);
        }

        std::stringstream ss;
        ss << "熵值: " << std::fixed << std::setprecision(3) << entropy;
        if (entropy < 3.0) ss << " (低熵 - 可能为文本)";
        else if (entropy < 6.0) ss << " (中熵)";
        else ss << " (高熵 - 可能为加密或随机数据)";

        return ss.str();
    }

    std::string extractStrings(const std::string& data, const std::string& options) {
        int minLength = options.empty() ? 4 : std::stoi(options);
        std::stringstream ss;
        std::string current;

        for (char c : data) {
            if (c >= 32 && c <= 126) {
                current += c;
            }
            else {
                if (current.length() >= minLength) {
                    ss << current << "\n";
                }
                current.clear();
            }
        }

        if (current.length() >= minLength) {
            ss << current << "\n";
        }

        return ss.str().empty() ? "未找到可打印字符串" : ss.str();
    }

    std::string toHex(const std::string& data) {
        std::stringstream ss;
        for (size_t i = 0; i < std::min(data.size(), (size_t)256); ++i) {
            if (i % 16 == 0 && i > 0) ss << "\n";
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)data[i] << " ";
        }
        if (data.size() > 256) {
            ss << "\n... (截断，共 " << data.size() << " 字节)";
        }
        return ss.str();
    }

    std::string findPatterns(const std::string& data) {
        std::stringstream ss;
        std::map<std::string, int> patterns;

        // 查找重复模式
        for (size_t len = 2; len <= 8; ++len) {
            for (size_t i = 0; i + len <= data.size(); ++i) {
                std::string pattern = data.substr(i, len);
                patterns[pattern]++;
            }
        }

        // 找出重复最多的模式
        std::vector<std::pair<std::string, int>> sorted;
        for (const auto& p : patterns) {
            if (p.second > 1 && p.first.length() >= 2) {
                sorted.push_back(p);
            }
        }

        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        ss << "发现的重复模式:\n";
        for (size_t i = 0; i < std::min((size_t)10, sorted.size()); ++i) {
            ss << "  \"" << sorted[i].first << "\" 出现 " << sorted[i].second << " 次\n";
        }

        return ss.str();
    }
};