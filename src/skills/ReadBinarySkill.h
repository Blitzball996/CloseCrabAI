#pragma once
#include "Skill.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

class ReadBinarySkill : public Skill {
public:
    std::string getName() const override { return "read_binary"; }

    std::string getDescription() const override {
        return "读取二进制文件并分析内容。参数: file_path (文件路径), format (hex/ascii/analysis), offset (起始位置), length (读取长度)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"file_path", "文件路径", "string", true},
            {"format", "输出格式: hex/ascii/analysis", "string", false},
            {"offset", "起始偏移(字节)", "int", false},
            {"length", "读取长度(字节)", "int", false}
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
        std::string format = params.count("format") ? params.at("format") : "analysis";
        size_t offset = params.count("offset") ? std::stoul(params.at("offset")) : 0;
        size_t length = params.count("length") ? std::stoul(params.at("length")) : 256;

        return readBinary(filePath, format, offset, length);
    }

private:
    std::string readBinary(const std::string& filePath, const std::string& format,
        size_t offset, size_t length) {
        // 打开文件
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            return "无法打开文件: " + filePath;
        }

        // 获取文件大小
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(offset, std::ios::beg);

        if (offset >= fileSize) {
            return "偏移量超出文件大小";
        }

        // 读取数据
        size_t readLen = std::min(length, fileSize - offset);
        std::vector<unsigned char> buffer(readLen);
        file.read(reinterpret_cast<char*>(buffer.data()), readLen);
        file.close();

        std::stringstream result;

        if (format == "hex") {
            // 十六进制输出
            for (size_t i = 0; i < buffer.size(); ++i) {
                if (i % 16 == 0) {
                    if (i > 0) result << "\n";
                    result << std::hex << std::setw(8) << std::setfill('0') << (offset + i) << ": ";
                }
                result << std::hex << std::setw(2) << std::setfill('0') << (int)buffer[i] << " ";
            }
        }
        else if (format == "ascii") {
            // ASCII 输出
            for (unsigned char c : buffer) {
                if (c >= 32 && c <= 126) {
                    result << c;
                }
                else {
                    result << '.';
                }
            }
        }
        else {
            // 分析模式：检测文件类型、熵值、模式
            result << analyzeBinary(buffer, filePath, fileSize);
        }

        return result.str();
    }

    std::string analyzeBinary(const std::vector<unsigned char>& data,
        const std::string& filePath, size_t fileSize) {
        std::stringstream ss;

        // 文件基本信息
        ss << "📁 文件: " << filePath << "\n";
        ss << "📦 大小: " << fileSize << " 字节";
        if (fileSize > 1024 * 1024) {
            ss << " (" << std::fixed << std::setprecision(2) << (fileSize / (1024.0 * 1024.0)) << " MB)";
        }
        else if (fileSize > 1024) {
            ss << " (" << std::fixed << std::setprecision(2) << (fileSize / 1024.0) << " KB)";
        }
        ss << "\n";

        // 检测文件头（魔数）
        ss << "\n🔍 文件头检测:\n";
        if (data.size() >= 4) {
            uint32_t magic = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
            ss << "  魔数: 0x" << std::hex << std::setw(8) << std::setfill('0') << magic << "\n";

            // 常见文件类型检测
            if (data[0] == 0x4D && data[1] == 0x5A) {
                ss << "  类型: Windows PE 可执行文件 (.exe/.dll)\n";
            }
            else if (data[0] == 0x7F && data[1] == 0x45 && data[2] == 0x4C && data[3] == 0x46) {
                ss << "  类型: ELF 可执行文件 (Linux)\n";
            }
            else if (data[0] == 0x50 && data[1] == 0x4B) {
                ss << "  类型: ZIP 压缩包 (.zip/.jar/.docx)\n";
            }
            else if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
                ss << "  类型: PNG 图片\n";
            }
            else if (data[0] == 0xFF && data[1] == 0xD8) {
                ss << "  类型: JPEG 图片\n";
            }
            else if (data[0] == 0x25 && data[1] == 0x50 && data[2] == 0x44 && data[3] == 0x46) {
                ss << "  类型: PDF 文档\n";
            }
            else {
                ss << "  类型: 未知或原始二进制\n";
            }
        }

        // 熵值分析（衡量随机性）
        ss << "\n📊 熵值分析:\n";
        double entropy = calculateEntropy(data);
        ss << "  信息熵: " << std::fixed << std::setprecision(3) << entropy;
        if (entropy < 3.0) {
            ss << " (低熵 - 可能是文本或压缩数据)\n";
        }
        else if (entropy < 6.0) {
            ss << " (中熵 - 可能是加密或混合数据)\n";
        }
        else {
            ss << " (高熵 - 可能是加密或随机数据)\n";
        }

        // 可打印字符比例
        int printable = 0;
        for (unsigned char c : data) {
            if (c >= 32 && c <= 126) printable++;
        }
        double printableRatio = (double)printable / data.size() * 100;
        ss << "  可打印字符: " << std::fixed << std::setprecision(1) << printableRatio << "%\n";

        // 检测是否可能是文本文件
        if (printableRatio > 80) {
            ss << "\n💡 建议: 文件可能是文本文件，可用 read_file 技能读取\n";
        }

        return ss.str();
    }

    double calculateEntropy(const std::vector<unsigned char>& data) {
        if (data.empty()) return 0.0;

        std::map<unsigned char, int> freq;
        for (unsigned char c : data) {
            freq[c]++;
        }

        double entropy = 0.0;
        for (const auto& pair : freq) {
            double p = (double)pair.second / data.size();
            entropy -= p * log2(p);
        }
        return entropy;
    }
};