#pragma once
#include "Skill.h"
#include <windows.h>
#include <gdiplus.h>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>

#pragma comment(lib, "gdiplus.lib")

class ScreenshotSkill : public Skill {
public:
    std::string getName() const override { return "screenshot"; }

    std::string getDescription() const override {
        return "截取屏幕截图。参数: file_path (可选，保存路径，默认保存到桌面)";
    }

    std::vector<SkillParameter> getParameters() const override {
        return {
            {"file_path", "保存路径（可选）", "string", false}
        };
    }

    bool needsConfirmation() const override { return false; }
    std::string getCategory() const override { return "system"; }
    PermissionLevel getPermissionLevel() const override { return PermissionLevel::NORMAL; }

    std::string execute(const std::map<std::string, std::string>& params) override {
        auto it = params.find("file_path");
        std::string filePath;

        if (it != params.end() && !it->second.empty()) {
            filePath = it->second;
        }
        else {
            // 默认保存到桌面，文件名包含时间戳
            char* desktop = nullptr;
            _dupenv_s(&desktop, nullptr, "USERPROFILE");
            if (desktop) {
                filePath = std::string(desktop) + "\\Desktop\\screenshot_";
                free(desktop);
            }
            else {
                filePath = "screenshot_";
            }

            auto now = std::time(nullptr);
            auto tm = *std::localtime(&now);
            std::stringstream ss;
            ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
            filePath += ss.str() + ".png";
        }

        return captureScreen(filePath);
    }

private:
    std::string captureScreen(const std::string& outputPath) {
        // 初始化 GDI+
        Gdiplus::GdiplusStartupInput gdiplusStartupInput;
        ULONG_PTR gdiplusToken;
        Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);

        // 获取屏幕尺寸
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);

        // 获取屏幕 DC
        HDC hdcScreen = GetDC(NULL);
        HDC hdcMem = CreateCompatibleDC(hdcScreen);
        HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, screenWidth, screenHeight);
        SelectObject(hdcMem, hBitmap);

        // 复制屏幕内容
        BitBlt(hdcMem, 0, 0, screenWidth, screenHeight, hdcScreen, 0, 0, SRCCOPY);

        // 转换为 GDI+ Bitmap
        Gdiplus::Bitmap bitmap(hBitmap, nullptr);

        // 保存为 PNG
        CLSID pngClsid;
        GetEncoderClsid(L"image/png", &pngClsid);

        std::wstring wPath(outputPath.begin(), outputPath.end());
        Gdiplus::Status status = bitmap.Save(wPath.c_str(), &pngClsid, nullptr);

        // 清理
        DeleteObject(hBitmap);
        DeleteDC(hdcMem);
        ReleaseDC(NULL, hdcScreen);
        Gdiplus::GdiplusShutdown(gdiplusToken);

        if (status == Gdiplus::Ok) {
            return "截图已保存到: " + outputPath;
        }
        else {
            return "截图保存失败: " + std::to_string(status);
        }
    }

    int GetEncoderClsid(const wchar_t* format, CLSID* pClsid) {
        UINT num = 0;
        UINT size = 0;
        Gdiplus::GetImageEncodersSize(&num, &size);
        if (size == 0) return -1;

        Gdiplus::ImageCodecInfo* pImageCodecInfo = (Gdiplus::ImageCodecInfo*)malloc(size);
        if (pImageCodecInfo == nullptr) return -1;

        Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);

        for (UINT i = 0; i < num; ++i) {
            if (wcscmp(pImageCodecInfo[i].MimeType, format) == 0) {
                *pClsid = pImageCodecInfo[i].Clsid;
                free(pImageCodecInfo);
                return i;
            }
        }

        free(pImageCodecInfo);
        return -1;
    }
};