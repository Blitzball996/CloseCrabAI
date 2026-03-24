#pragma once
#include "Skill.h"
#include <windows.h>
#include <psapi.h>
#include <sstream>
#include <iomanip>

#pragma comment(lib, "psapi.lib")

class SystemInfoSkill : public Skill {
public:
    std::string getName() const override { return "system_info"; }

    std::string getDescription() const override {
        return "获取系统信息（CPU、内存、磁盘等）。参数: type (可选，cpu/memory/disk/all)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"type", "信息类型 (cpu/memory/disk/all)", "string", false}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "system"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::SAFE; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("type");
        std::string type = (it != params.end()) ? it->second : "all";

        std::string result;
        if (type == "cpu" || type == "all") {
            result += getCpuInfo();
        }
        if (type == "memory" || type == "all") {
            result += getMemoryInfo();
        }
        if (type == "disk" || type == "all") {
            result += getDiskInfo();
        }

        return result.empty() ? "无法获取系统信息" : result;
    }

private:
    std::string getCpuInfo() {
        std::stringstream ss;
        ss << "📊 CPU信息:\n";

        // 使用宽字符版本
        HKEY hKey;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
            L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            wchar_t processorName[256];
            DWORD size = sizeof(processorName);
            if (RegQueryValueExW(hKey, L"ProcessorNameString", nullptr, nullptr,
                (LPBYTE)processorName, &size) == ERROR_SUCCESS) {
                std::wstring wname(processorName);
                std::string name(wname.begin(), wname.end());
                ss << "  型号: " << name << "\n";
            }
            RegCloseKey(hKey);
        }

        // 获取 CPU 核心数
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        ss << "  核心数: " << sysInfo.dwNumberOfProcessors << "\n";

        // 获取 CPU 使用率
        static ULONGLONG lastIdle = 0, lastKernel = 0, lastUser = 0;
        FILETIME idleTime, kernelTime, userTime;
        if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
            ULONGLONG idle = ((ULONGLONG)idleTime.dwHighDateTime << 32) | idleTime.dwLowDateTime;
            ULONGLONG kernel = ((ULONGLONG)kernelTime.dwHighDateTime << 32) | kernelTime.dwLowDateTime;
            ULONGLONG user = ((ULONGLONG)userTime.dwHighDateTime << 32) | userTime.dwLowDateTime;

            if (lastIdle != 0) {
                ULONGLONG total = (kernel - lastKernel) + (user - lastUser);
                ULONGLONG idleDiff = idle - lastIdle;
                if (total > 0) {
                    int usage = (int)(100 - (idleDiff * 100.0 / total));
                    ss << "  使用率: " << usage << "%\n";
                }
            }
            lastIdle = idle;
            lastKernel = kernel;
            lastUser = user;
        }

        ss << "\n";
        return ss.str();
    }

    std::string getMemoryInfo() {
        std::stringstream ss;
        ss << "💾 内存信息:\n";

        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            ss << "  总量: " << (memStatus.ullTotalPhys / (1024 * 1024 * 1024)) << " GB\n";
            ss << "  可用: " << (memStatus.ullAvailPhys / (1024 * 1024 * 1024)) << " GB\n";
            ss << "  使用率: " << memStatus.dwMemoryLoad << "%\n";
        }

        ss << "\n";
        return ss.str();
    }

    std::string getDiskInfo() {
        std::stringstream ss;
        ss << "💿 磁盘信息:\n";

        DWORD drives = GetLogicalDrives();
        for (char drive = 'A'; drive <= 'Z'; ++drive) {
            if (drives & 1) {
                std::string root = std::string(1, drive) + ":\\";
                // 使用 ANSI 版本
                ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;
                if (GetDiskFreeSpaceExA(root.c_str(), &freeBytes, &totalBytes, &totalFreeBytes)) {
                    ULONGLONG totalGB = totalBytes.QuadPart / (1024 * 1024 * 1024);
                    ULONGLONG freeGB = freeBytes.QuadPart / (1024 * 1024 * 1024);
                    if (totalGB > 0) {
                        int usage = (int)((totalGB - freeGB) * 100.0 / totalGB);
                        ss << "  " << drive << ": " << totalGB << " GB, 已用 " << usage << "%\n";
                    }
                }
            }
            drives >>= 1;
        }

        ss << "\n";
        return ss.str();
    }
};