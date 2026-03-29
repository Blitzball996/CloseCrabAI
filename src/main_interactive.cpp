#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include "config/Config.h"
#include "core/SessionManager.h"
#include "memory/MemorySystem.h"
#include "core/ThreadPool.h"
#include "llm/LLMEngine.h"
#include "network/WebSocketServer.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#include "skills/SkillManager.h"
#include "skills/OpenAppSkill.h"
#include "skills/ReadFileSkill.h"
#include "skills/ReadBinarySkill.h"
#include "skills/WriteFileSkill.h"
#include "skills/ExecuteCommandSkill.h"
#include "skills/WebSearchSkill.h"
#include "skills/ScreenshotSkill.h"
#include "skills/SystemInfoSkill.h"
#include "skills/ClipboardSkill.h"
#include "skills/WeatherSkill.h"
#include "security/Sandbox.h"
#include "network/HttpServer.h"
#include "rag/RAGManager.h"
#include "ssd/SSDExpertStreamer.h"

bool g_llama_log_enabled = false;
bool running = true;

// 获取 exe 所在目录
std::string getExePath() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string path(buffer);
    return path.substr(0, path.find_last_of("\\/"));
}

std::string trim(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }
    auto end = str.end();
    do {
        --end;
    } while (end != start && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(start, end + 1);
}

// GBK 转 UTF-8
std::string gbkToUtf8(const std::string& gbkStr) {
    if (gbkStr.empty()) return "";

    // GBK -> UTF-16
    int wideSize = MultiByteToWideChar(936, 0, gbkStr.c_str(), -1, nullptr, 0);
    if (wideSize == 0) return "";

    std::wstring wideStr(wideSize, 0);
    MultiByteToWideChar(936, 0, gbkStr.c_str(), -1, &wideStr[0], wideSize);

    // UTF-16 -> UTF-8
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, wideStr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (utf8Size == 0) return "";

    std::string utf8Str(utf8Size, 0);
    WideCharToMultiByte(CP_UTF8, 0, wideStr.c_str(), -1, &utf8Str[0], utf8Size, nullptr, nullptr);

    // 去除末尾的 null 字符
    while (!utf8Str.empty() && utf8Str.back() == '\0') {
        utf8Str.pop_back();
    }

    return utf8Str;
}

// UTF-8 转 GBK
std::string utf8ToGbk(const std::string& utf8Str) {
    if (utf8Str.empty()) return "";

    // UTF-8 -> UTF-16
    int wideSize = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, nullptr, 0);
    if (wideSize == 0) return "";

    std::wstring wideStr(wideSize, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, &wideStr[0], wideSize);

    // UTF-16 -> GBK
    int gbkSize = WideCharToMultiByte(936, 0, wideStr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (gbkSize == 0) return "";

    std::string gbkStr(gbkSize, 0);
    WideCharToMultiByte(936, 0, wideStr.c_str(), -1, &gbkStr[0], gbkSize, nullptr, nullptr);

    // 去除末尾的 null 字符
    while (!gbkStr.empty() && gbkStr.back() == '\0') {
        gbkStr.pop_back();
    }

    return gbkStr;
}

// ============================================
// Qwen 格式辅助函数
// ============================================

// 构建系统消息
std::string buildSystemMessage(const std::string& content) {
    return "<|im_start|>system\n" + content + "<|im_end|>\n";
}

// 构建用户消息
std::string buildUserMessage(const std::string& content) {
    return "<|im_start|>user\n" + content + "<|im_end|>\n";
}

// 构建助手消息
std::string buildAssistantMessage(const std::string& content) {
    return "<|im_start|>assistant\n" + content + "<|im_end|>\n";
}

// 解析 AI 回复中的技能调用
std::string parseAndExecuteSkill(const std::string& response) {
    // 查找 SKILL: 标记
    size_t skillPos = response.find("SKILL:");
    if (skillPos == std::string::npos) {
        return response;
    }

    // 提取技能部分
    std::string skillPart = response.substr(skillPos);

    // 解析技能名
    size_t nameStart = skillPart.find(':') + 1;
    size_t nameEnd = skillPart.find('\n', nameStart);
    if (nameEnd == std::string::npos) {
        nameEnd = skillPart.length();
    }
    std::string skillName = skillPart.substr(nameStart, nameEnd - nameStart);
    skillName.erase(0, skillName.find_first_not_of(" \t"));
    skillName.erase(skillName.find_last_not_of(" \t") + 1);

    // 查找 PARAMS:
    size_t paramsPos = skillPart.find("PARAMS:");
    std::map<std::string, std::string> params;

    if (paramsPos != std::string::npos) {
        std::string paramsStr = skillPart.substr(paramsPos + 7);
        size_t paramsEnd = paramsStr.find('\n');
        if (paramsEnd != std::string::npos) {
            paramsStr = paramsStr.substr(0, paramsEnd);
        }

        // 解析参数 (格式: key=value, key2=value2)
        size_t start = 0;
        while (start < paramsStr.length()) {
            size_t eqPos = paramsStr.find('=', start);
            if (eqPos == std::string::npos) break;

            std::string key = paramsStr.substr(start, eqPos - start);
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);

            size_t commaPos = paramsStr.find(',', eqPos + 1);
            if (commaPos == std::string::npos) {
                commaPos = paramsStr.length();
            }

            std::string value = paramsStr.substr(eqPos + 1, commaPos - eqPos - 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            params[key] = value;
            start = commaPos + 1;
        }
    }

    // 执行技能
    auto& skillManager = SkillManager::getInstance();
    std::string result = skillManager.executeSkill(skillName, params);

    // 返回格式化的结果
    return "\n[Skill Result]\n" + result + "\n[End Skill Result]\n";
}

// ============================================
// UI 函数
// ============================================

// ============================================================
// 修复 Windows 控制台输入字符数限制
// ============================================================
//
// 用下面的函数替换 main_interactive.cpp 中的 getUserInput()
// （原来在第 194-229 行）
//
// 原来的问题：std::getline + cmd.exe 最多输入约 4096 字符
// 修复后：使用 ReadConsoleW 直接读取，支持最多 65536 字符
//         并且直接输出 UTF-8，不再需要 gbkToUtf8 转换
// ============================================================

std::string getUserInput() {
    std::cout << "\n\033[32mYou:\033[0m " << std::flush;

#ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    if (hStdin == INVALID_HANDLE_VALUE) {
        std::string input;
        std::getline(std::cin, input);
        return gbkToUtf8(input);
    }

    DWORD originalMode = 0;
    GetConsoleMode(hStdin, &originalMode);
    SetConsoleMode(hStdin, ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);

    std::string result;
    bool firstLine = true;

    while (true) {
        const DWORD bufSize = 65536;
        std::vector<wchar_t> wBuf(bufSize);
        DWORD charsRead = 0;

        BOOL ok = ReadConsoleW(hStdin, wBuf.data(), bufSize - 1, &charsRead, NULL);
        if (!ok || charsRead == 0) break;

        // 去掉末尾 \r\n
        while (charsRead > 0 && (wBuf[charsRead - 1] == L'\r' || wBuf[charsRead - 1] == L'\n')) {
            charsRead--;
        }
        wBuf[charsRead] = L'\0';

        // UTF-16 → UTF-8
        int utf8Size = WideCharToMultiByte(CP_UTF8, 0, wBuf.data(), (int)charsRead,
            nullptr, 0, nullptr, nullptr);
        if (utf8Size > 0) {
            std::string line(utf8Size, '\0');
            WideCharToMultiByte(CP_UTF8, 0, wBuf.data(), (int)charsRead,
                &line[0], utf8Size, nullptr, nullptr);
            while (!line.empty() && line.back() == '\0') line.pop_back();

            if (!firstLine) result += "\n";
            result += line;
            firstLine = false;
        }

        // 关键：检查控制台输入缓冲区里是否还有待处理的数据
        // 如果有 = 用户在粘贴多行文本，继续读取
        // 如果没有 = 用户打完了，返回结果
        DWORD eventsAvailable = 0;
        GetNumberOfConsoleInputEvents(hStdin, &eventsAvailable);
        if (eventsAvailable == 0) {
            // 等一小段时间，看看是不是粘贴还没完成
            Sleep(50);  // 50ms
            GetNumberOfConsoleInputEvents(hStdin, &eventsAvailable);
            if (eventsAvailable == 0) {
                break;  // 确实没有更多输入了，返回
            }
        }
    }

    SetConsoleMode(hStdin, originalMode);

    // trim
    size_t start = result.find_first_not_of(" \t\r\n");
    size_t end = result.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);

#else
    std::string input;
    std::getline(std::cin, input);
    return input;
#endif
}

void printHelp() {
    std::cout << "\n\033[33m=== Commands ===\033[0m" << std::endl;
    std::cout << "  /help      - Show this help" << std::endl;
    std::cout << "  /clear     - Clear conversation history" << std::endl;
    std::cout << "  /new       - Start a new session" << std::endl;
    std::cout << "  /history   - Show conversation history" << std::endl;
    std::cout << "  /skillmode - Show skill mode status" << std::endl;
    std::cout << "  /skills    - List available skills" << std::endl;
    std::cout << "  /sandbox   - Show sandbox status" << std::endl;
    std::cout << "  /sandbox disable  - Disable sandbox" << std::endl;
    std::cout << "  /sandbox ask      - Ask before dangerous actions" << std::endl;
    std::cout << "  /sandbox auto     - Auto block dangerous actions" << std::endl;
    std::cout << "  /sandbox trusted  - Trusted mode (log only)" << std::endl;
    std::cout << "  /rag       - Show rag status" << std::endl;
    std::cout << "  /cpu-moe <N> - Force CPU expert layers to run on cpu" << std::endl;
    std::cout << "  /ssd       - Show SSD streaming status" << std::endl;
    std::cout << "  /audit     - Show audit log" << std::endl;
    std::cout << "  /quit      - Exit the program" << std::endl;
    std::cout << "\nJust type your message to chat with AI." << std::endl;
    std::cout << "================================\033[0m" << std::endl;
}

void showSandboxStatus() {
    auto& sandbox = Sandbox::getInstance();
    std::cout << "\n\033[33m=== Sandbox Status ===\033[0m" << std::endl;
    std::string modeStr;
    switch (sandbox.getMode()) {
    case Sandbox::Mode::DISABLED: modeStr = "DISABLED (no protection)"; break;
    case Sandbox::Mode::ASK: modeStr = "ASK (ask before dangerous)"; break;
    case Sandbox::Mode::AUTO: modeStr = "AUTO (auto block dangerous)"; break;
    case Sandbox::Mode::TRUSTED: modeStr = "TRUSTED (log only)"; break;
    }
    std::cout << "Mode: " << modeStr << std::endl;
    std::cout << "Audit log entries: " << sandbox.getAuditLog().size() << std::endl;
    std::cout << "\033[33m========================\033[0m" << std::endl;
}

void showAuditLog() {
    auto& sandbox = Sandbox::getInstance();
    auto log = sandbox.getAuditLog();
    if (log.empty()) {
        std::cout << "No audit log entries." << std::endl;
        return;
    }
    std::cout << "\n\033[33m=== Audit Log ===\033[0m" << std::endl;
    for (const auto& entry : log) {
        std::cout << "  " << entry << std::endl;
    }
    std::cout << "\033[33m==================\033[0m" << std::endl;
}

void showHistory(MemorySystem& memory, const std::string& sessionId) {
    auto memories = memory.getRecentMemories(sessionId, 50);
    if (memories.empty()) {
        std::cout << "No conversation history." << std::endl;
        return;
    }

    std::cout << "\n\033[33m=== Conversation History ===\033[0m" << std::endl;
    for (const auto& mem : memories) {
        if (mem.role == "user") {
            std::cout << "\033[32mYou:\033[0m " << mem.content << std::endl;
        }
        else if (mem.role == "assistant") {
            std::cout << "\033[34mAI:\033[0m " << mem.content << std::endl;
        }
    }
    std::cout << "\033[33m============================\033[0m" << std::endl;
}

void showSkills() {
    auto& skillManager = SkillManager::getInstance();
    std::cout << "\n\033[33m=== Available Skills ===\033[0m" << std::endl;
    std::cout << skillManager.getSkillsDescription();
    std::cout << "\033[33m========================\033[0m" << std::endl;
}

// ============================================
// 主函数
// ============================================

int main(int argc, char* argv[]) {
    // ========== 最佳控制台编码设置 ==========
#ifdef _WIN32
    // 启用控制台 ANSI 颜色
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif

    // 初始化 Winsock（在程序最开头）
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        std::cerr << "WSAStartup failed: " << result << std::endl;
        return 1;
    }
    spdlog::info("Winsock initialized");

    CLI::App app{ "CloseCrab - AI Execution Engine (Interactive Mode)" };

    std::string configFile = "config/config.yaml";
    bool verbose = false;
    bool noWebSocket = false;
    int webSocketPort = 9001;
    int httpPort = 8080;        // 新增
    bool noHttp = false;
    std::string modelPathOverride;
    std::string sessionIdOverride;

    app.add_option("-c,--config", configFile, "Config file path");
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");
    app.add_flag("--no-ws", noWebSocket, "Disable WebSocket server");
    app.add_option("--ws-port", webSocketPort, "WebSocket server port (default: 9001)");
    app.add_option("-m,--model", modelPathOverride, "Override model path");
    app.add_option("-s,--session", sessionIdOverride, "Resume existing session");
    app.add_option("--http-port", httpPort, "HTTP API port (default: 8080)");
    app.add_flag("--no-http", noHttp, "Disable HTTP API server");
    int n_cpu_moe = 0;   // 定义变量
    app.add_option("--n-cpu-moe", n_cpu_moe, "Number of MoE expert layers to put on CPU (0 = all on GPU)");
    // ← 新增：SSD streaming 参数
    std::string ssdExpertDir = "";
    size_t ssdCacheMB = 32768;       // 默认 32GB RAM 缓存
    size_t ssdGpuCacheMB = 4096;     // 默认 4GB GPU 缓存
    app.add_option("--expert-dir", ssdExpertDir, "Packed expert files directory for SSD streaming");
    app.add_option("--ssd-cache-mb", ssdCacheMB, "SSD streamer RAM cache size in MB (default: 32768)");
    app.add_option("--ssd-gpu-cache-mb", ssdGpuCacheMB, "SSD streamer GPU cache size in MB (default: 4096)");

    try {
        CLI11_PARSE(app, argc, argv);
    }
    catch (const CLI::ParseError& e) {
        WSACleanup();
        return app.exit(e);
    }

    // 获取 exe 所在目录
    auto getExePath = []() -> std::string {
        char buffer[MAX_PATH];
        GetModuleFileNameA(NULL, buffer, MAX_PATH);
        std::string path(buffer);
        return path.substr(0, path.find_last_of("\\/"));
        };

    // 构建配置文件路径
    std::string exeDir = getExePath();
    std::string configPath = exeDir + "\\config\\config.yaml";

    // 如果 exe 目录下没有，尝试原路径
    if (!std::filesystem::exists(configPath)) {
        configPath = configFile;  // configFile 是命令行参数指定的路径
    }

    // 加载配置
    auto& config = Config::getInstance();
    if (!config.load(configPath)) {
        spdlog::error("Failed to load config, exiting...");
        WSACleanup();
        return 1;
    }

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
        spdlog::debug("Verbose mode enabled");
    }
    else {
        std::string logLevel = config.getString("logging.level", "info");
        if (logLevel == "debug") spdlog::set_level(spdlog::level::debug);
        else if (logLevel == "info") spdlog::set_level(spdlog::level::info);
        else if (logLevel == "warn") spdlog::set_level(spdlog::level::warn);
        else if (logLevel == "error") spdlog::set_level(spdlog::level::err);
    }

    std::string dbPath = config.getString("database.path", "data/closecrab.db");
    std::string modelPath = modelPathOverride.empty()
        ? config.getString("llm.model_path", "models/Qwen2.5-7B-Instruct-Uncensored.Q4_K_M.gguf")
        : modelPathOverride;

    // 添加调试输出
    spdlog::info("DEBUG: model_path from config = {}", config.getString("llm.model_path", "NOT_FOUND"));
    spdlog::info("DEBUG: final modelPath = {}", modelPath);

    int maxTokens = config.getInt("llm.max_tokens", 512);
    float temperature = static_cast<float>(config.getDouble("llm.temperature", 0.7));

    // ========== 在这里插入自动检测 MoE 的代码 ==========
    if (app.count("--n-cpu-moe") == 0) {
        std::string modelFilename = std::filesystem::path(modelPath).filename().string();
        std::string lowerFilename = modelFilename;
        for (auto& c : lowerFilename) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lowerFilename.find("moe") != std::string::npos) {
            n_cpu_moe = 4;
            spdlog::info("Detected MoE model ({}), automatically setting CPU MoE layers to {}",
                modelFilename, n_cpu_moe);
        }
    }
    // ===============================================

    spdlog::info("========================================");
    spdlog::info("CloseCrab Interactive Mode");
    spdlog::info("========================================");
    spdlog::info("Database: {}", dbPath);
    spdlog::info("Model: {}", modelPath);
    if (!noWebSocket) {
        spdlog::info("WebSocket: enabled on port {}", webSocketPort);
    }
    else {
        spdlog::info("WebSocket: disabled");
    }

    // 初始化组件
    SessionManager sessionManager(dbPath);

    std::string sessionId;
    if (!sessionIdOverride.empty() && sessionManager.getSession(sessionIdOverride)) {
        sessionId = sessionIdOverride;
        spdlog::info("Resumed session: {}", sessionId);
    }
    else {
        sessionId = sessionManager.createSession("interactive_user");
        spdlog::info("Created new session: {}", sessionId);
    }

    MemorySystem memorySystem(dbPath);
    auto& threadPool = ThreadPool::getInstance();
    spdlog::info("Thread pool size: {}", threadPool.getThreadCount());

    spdlog::info("Loading model... This may take a moment.");
    std::unique_ptr<LLMEngine> llm;
    llm = std::make_unique<LLMEngine>(modelPath, n_cpu_moe);
    if (!llm->isLoaded()) {
        spdlog::error("Failed to load model!");
        WSACleanup();
        return 1;
    }

    spdlog::info("Model loaded successfully!");

    // ← 新增：初始化 SSD Expert Streaming
    if (!ssdExpertDir.empty()) {
        spdlog::info("Initializing SSD Expert Streaming from: {}", ssdExpertDir);
        if (llm->initSSDStreaming(ssdExpertDir, ssdCacheMB, ssdGpuCacheMB)) {
            spdlog::info("SSD Expert Streaming: ENABLED");
        }
        else {
            spdlog::warn("SSD Expert Streaming: initialization FAILED (continuing without)");
        }
    }
    else {
        // 自动检测：如果 packed_experts/ 目录存在，自动启用
        std::string autoExpertDir = exeDir + "\\packed_experts";
        if (std::filesystem::exists(autoExpertDir)) {
            spdlog::info("Auto-detected packed_experts directory: {}", autoExpertDir);
            if (llm->initSSDStreaming(autoExpertDir, ssdCacheMB, ssdGpuCacheMB)) {
                spdlog::info("SSD Expert Streaming: auto-ENABLED");
            }
        }
    }

    /*
    // ========== 中文测试代码 ==========
    spdlog::info("Testing Chinese language support...");

    // 构建测试 prompt（全部使用 UTF-8）
    std::string testSystemMsg = buildSystemMessage("You are a helpful assistant. Respond in Chinese.");
    std::string testUserMsg = buildUserMessage("你好，请用中文回答，说一句话即可");
    std::string testPrompt = testSystemMsg + testUserMsg + "<|im_start|>assistant\n";

    spdlog::debug("Test prompt: {}", testPrompt);

    std::string testResponse = llm.generateRaw(testPrompt, 100, 0.7);

    // 将模型输出转为 GBK 再显示到控制台（避免 spdlog 输出乱码）
    std::string gbkResponse = utf8ToGbk(testResponse);
    std::cout << "Chinese test response: " << gbkResponse << std::endl;

    if (testResponse.empty()) {
        spdlog::error("Model generated empty response!");
    }
    else if (testResponse.find("你好") != std::string::npos ||
        testResponse.find("好") != std::string::npos ||
        testResponse.find("您") != std::string::npos) {
        spdlog::info("✓ Chinese support confirmed!");
    }
    else {
        spdlog::warn("✗ Model may not support Chinese properly!");
        spdlog::warn("Response was: {}", testResponse); // 这里仍可能乱码，但你可以忽略或也转码
        spdlog::warn("Please ensure you're using a Chinese-capable model like Qwen");
    }
    // ========== 测试代码结束 ==========
    */

    // 注册技能
    auto& skillManager = SkillManager::getInstance();
    skillManager.registerSkill(std::make_unique<OpenAppSkill>());
    skillManager.registerSkill(std::make_unique<ReadFileSkill>());
    skillManager.registerSkill(std::make_unique<ReadBinarySkill>());
    skillManager.registerSkill(std::make_unique<WriteFileSkill>());
    skillManager.registerSkill(std::make_unique<ExecuteCommandSkill>());
    skillManager.registerSkill(std::make_unique<WebSearchSkill>());
    skillManager.registerSkill(std::make_unique<ScreenshotSkill>());
    skillManager.registerSkill(std::make_unique<SystemInfoSkill>());
    skillManager.registerSkill(std::make_unique<ClipboardSkill>());
    skillManager.registerSkill(std::make_unique<WeatherSkill>());
    spdlog::info("Registered {} skills", skillManager.listSkills().size());

    // ====================设置初始SkillMode========================
    SkillManager::getInstance().setMode(SkillMode::CHAT_ONLY);

    auto& rag = RAGManager::getInstance();
    if (!rag.init("data/vectors.db")) {
        spdlog::warn("RAG initialization failed, will continue without RAG");
    }
    else {
        // RAG 默认已经禁用，不需要额外设置
        spdlog::info("RAG initialized but disabled by default. Use /rag enable to activate");
    }

    printHelp();
    showSkills();

    // 欢迎消息
    std::string welcomePrompt = "Hello! I am CloseCrab, your AI assistant. I can use skills to help you. Type /skills to see what I can do.";
    std::cout << "\n\033[34mAI:\033[0m " << welcomePrompt << std::endl;

    // ============================================
    // 启动 HTTP API 服务器
    // ============================================
    std::unique_ptr<HttpServer> httpServer;   // ← 先声明
    if (!noHttp) {
        httpServer = std::make_unique<HttpServer>(httpPort);
        httpServer->onChat([&](const std::string& message, const std::string& reqSessionId) -> std::string {
            std::string activeSessionId = reqSessionId.empty() ? sessionId : reqSessionId;
            memorySystem.addMemory(activeSessionId, "user", message);

            std::string systemContent =
                "You are CloseCrab, a helpful AI assistant. "
                "Respond in the same language as the user's question.";

            std::string fullPrompt = buildSystemMessage(systemContent);

            auto memories = memorySystem.getRecentMemories(activeSessionId, 20);
            for (const auto& mem : memories) {
                if (mem.role == "user") {
                    fullPrompt += buildUserMessage(mem.content);
                }
                else {
                    fullPrompt += buildAssistantMessage(mem.content);
                }
            }

            fullPrompt += buildUserMessage(message);
            fullPrompt += "<|im_start|>assistant\n";

            std::string response = llm->generateRaw(fullPrompt, maxTokens, temperature);

            if (!response.empty()) {
                memorySystem.addMemory(activeSessionId, "assistant", response);
            }

            return response;
            });
        httpServer->start();
        spdlog::info("HTTP API server started on http://localhost:{}", httpPort);
    }

    // ============================================
    // 启动 WebSocket 服务器
    // ============================================
    std::unique_ptr<WebSocketServer> wsServer;
    if (!noWebSocket) {
        wsServer = std::make_unique<WebSocketServer>(webSocketPort);

        // 构建系统提示（包含技能信息）
        std::string systemContent =
            "You are CloseCrab, a helpful AI assistant with skills.\n\n"
            "You can use skills by responding with:\n"
            "SKILL: skill_name\n"
            "PARAMS: param1=value1, param2=value2\n\n"
            "Available skills:\n" + skillManager.getSkillsDescription() + "\n"
            "When you need to perform an action, use the SKILL format. "
            "After the skill executes, you'll see the result. "
            "Respond in the same language as the user's question.";

        wsServer->onMessage([&](const std::string& clientId, const std::string& message) {
            if (message == "__connected__" || message == "__disconnected__") {
                return;
            }

            spdlog::info("WebSocket request from {}: {}", clientId, message);

            // 保存用户消息
            memorySystem.addMemory(sessionId, "user", message);

            // 构建完整 Prompt
            std::string fullPrompt = buildSystemMessage(systemContent);

            auto memories = memorySystem.getRecentMemories(sessionId, 20);
            for (const auto& mem : memories) {
                if (mem.role == "user") {
                    fullPrompt += buildUserMessage(mem.content);
                }
                else {
                    fullPrompt += buildAssistantMessage(mem.content);
                }
            }

            fullPrompt += buildUserMessage(message);
            fullPrompt += "<|im_start|>assistant\n";

            // 生成回复
            std::string response = llm->generateRaw(fullPrompt, maxTokens, temperature);

            // 解析并执行技能
            std::string finalResponse = parseAndExecuteSkill(response);

            // 保存助手回复
            if (!finalResponse.empty()) {
                memorySystem.addMemory(sessionId, "assistant", finalResponse);
            }

            // 发送回复
            wsServer->sendMessage(clientId, finalResponse);
            });

        wsServer->start();
        spdlog::info("WebSocket server started on ws://localhost:{}", webSocketPort);
    }

    // ============================================
    // 主对话循环
    // ============================================

    while (running) {
        // 1. 获取原始输入（GBK）
        std::string input = getUserInput();

        /*
        // ========== 调试输出 ==========
        spdlog::info("DEBUG: input = [{}]", input);
        spdlog::info("DEBUG: length = {}", input.length());
        spdlog::info("DEBUG: equals /skillmode = {}", input == "/skillmode");
        // ==============================

        
        // ========== 在这里添加详细的调试信息 ==========
        spdlog::info("=== User Input Debug ===");
        spdlog::info("Raw input length: {}", input.length());
        spdlog::info("Raw input: {}", input);

        // 打印每个字符的十六进制
        std::string hexStr;
        for (unsigned char c : input) {
            char buf[4];
            sprintf(buf, "%02X ", c);
            hexStr += buf;
        }
        spdlog::info("Hex: {}", hexStr);

        // 检测是否包含中文字符
        bool hasChinese = false;
        for (unsigned char c : input) {
            // UTF-8 中文编码范围：0xE4-0xE9 开头
            if (c >= 0xE4 && c <= 0xE9) {
                hasChinese = true;
                break;
            }
        }
        spdlog::info("Contains Chinese: {}", hasChinese);
        spdlog::info("=========================");
        // ========== 调试信息添加结束 ==========
        */

        // 处理命令
        if (input == "/quit" || input == "/exit") {
            std::cout << "Goodbye!" << std::endl;
            break;
        }
        if (input == "/help") {
            printHelp();
            continue;
        }
        if (input == "/skillmode") {
            auto& skillManager = SkillManager::getInstance();
            std::cout << "\n\033[33m=== Skill Mode ===\033[0m" << std::endl;
            std::cout << "Current mode: " << skillManager.getModeName() << std::endl;
            std::cout << "Commands:" << std::endl;
            std::cout << "  /skillmode          - Show skill mode status" << std::endl;
            std::cout << "  /skillmode auto     - AI decides when to use skills" << std::endl;
            std::cout << "  /skillmode chat     - Chat only, no skills" << std::endl;
            std::cout << "  /skillmode skill    - Skill only, no chat" << std::endl;
            std::cout << "  /skillmode ask      - Ask before executing skills" << std::endl;
            std::cout << "\033[33m===================\033[0m" << std::endl;
            continue;
        }

        if (input == "/skillmode chat") {
            SkillManager::getInstance().setMode(SkillMode::CHAT_ONLY);
            std::cout << "Skill mode: CHAT_ONLY - AI will only chat, no skills" << std::endl;
            continue;
        }

        if (input == "/skillmode auto") {
            SkillManager::getInstance().setMode(SkillMode::AUTO);
            std::cout << "Skill mode: AUTO - AI will decide when to use skills" << std::endl;
            continue;
        }

        if (input == "/skillmode skill") {
            SkillManager::getInstance().setMode(SkillMode::SKILL_ONLY);
            std::cout << "Skill mode: SKILL_ONLY - AI will only execute skills, no chat" << std::endl;
            continue;
        }

        if (input == "/skillmode ask") {
            SkillManager::getInstance().setMode(SkillMode::ASK);
            std::cout << "Skill mode: ASK - Will ask before executing skills" << std::endl;
            continue;
        }
        if (input == "/skills") {
            showSkills();
            continue;
        }
        if (input == "/clear") {
            memorySystem.clearMemories(sessionId);
            std::cout << "Conversation history cleared." << std::endl;
            continue;
        }
        if (input == "/new") {
            sessionId = sessionManager.createSession("interactive_user");
            spdlog::info("Created new session: {}", sessionId);
            std::cout << "Started new conversation session." << std::endl;
            continue;
        }
        if (input == "/history") {
            showHistory(memorySystem, sessionId);
            continue;
        }
        if (input == "/http") {   // 新增
            if (httpServer && httpServer->isRunning()) {
                std::cout << "HTTP API server running on http://localhost:" << httpPort << std::endl;
                std::cout << "  POST /chat - Send message" << std::endl;
                std::cout << "  GET /health - Health check" << std::endl;
            }
            else {
                std::cout << "HTTP API server disabled" << std::endl;
            }
            continue;
        }
        // ========== 新增：沙箱命令 ==========
        if (input == "/sandbox") {
            showSandboxStatus();
            continue;
        }
        if (input == "/sandbox disable") {
            Sandbox::getInstance().setMode(Sandbox::Mode::DISABLED);
            std::cout << "Sandbox disabled. All skills will execute directly." << std::endl;
            continue;
        }
        if (input == "/sandbox ask") {
            Sandbox::getInstance().setMode(Sandbox::Mode::ASK);
            std::cout << "Sandbox set to ASK mode. Will ask before dangerous actions." << std::endl;
            continue;
        }
        if (input == "/sandbox auto") {
            Sandbox::getInstance().setMode(Sandbox::Mode::AUTO);
            std::cout << "Sandbox set to AUTO mode. Dangerous actions will be blocked." << std::endl;
            continue;
        }
        if (input == "/sandbox trusted") {
            Sandbox::getInstance().setMode(Sandbox::Mode::TRUSTED);
            std::cout << "Sandbox set to TRUSTED mode. Actions logged but not blocked." << std::endl;
            continue;
        }
        if (input == "/audit") {
            showAuditLog();
            continue;
        }
        // ← 新增：SSD Streaming 命令
        if (input == "/ssd") {
            if (llm->isSSDStreamingEnabled()) {
                std::cout << "\n\033[33m=== SSD Expert Streaming ===\033[0m" << std::endl;
                std::cout << llm->getSSDStreamerStatus() << std::endl;
                std::cout << "\033[33m============================\033[0m" << std::endl;
            }
            else {
                std::cout << "SSD Expert Streaming: not initialized" << std::endl;
                std::cout << "Use --expert-dir <path> to enable" << std::endl;
            }
            continue;
        }
        // ========== 新增：RAG命令 ==========
        if (input == "/rag") {
            auto& rag = RAGManager::getInstance();
            std::cout << "\n\033[33m=== RAG Status ===\033[0m" << std::endl;
            // 显示启用状态
            if (rag.isEnabled()) {
                std::cout << "Status: \033[32mENABLED\033[0m" << std::endl;
            }
            else {
                std::cout << "Status: \033[31mDISABLED\033[0m" << std::endl;
            }
            std::cout << "Documents loaded: " << rag.getDocumentCount() << std::endl;
            std::cout << "\nCommands:" << std::endl;
            std::cout << "  /rag enable      - Enable RAG (use documents in responses)" << std::endl;
            std::cout << "  /rag disable     - Disable RAG" << std::endl;
            std::cout << "  /rag load <path> - Load documents from directory" << std::endl;
            std::cout << "  /rag clear       - Clear all documents" << std::endl;
            std::cout << "  /rag list        - List all documents" << std::endl;
            std::cout << "\033[33m==================\033[0m" << std::endl;
            continue;
        }

        // 新增：启用 RAG
        if (input == "/rag enable") {
            auto& rag = RAGManager::getInstance();
            rag.setEnabled(true);
            std::cout << "\033[32mRAG enabled.\033[0m Documents will now be used to enhance responses." << std::endl;
            continue;
        }

        // 新增：禁用 RAG
        if (input == "/rag disable") {
            auto& rag = RAGManager::getInstance();
            rag.setEnabled(false);
            std::cout << "\033[33mRAG disabled.\033[0m Responses will not include document context." << std::endl;
            continue;
        }

        // 修改其他 RAG 命令，添加提示
        if (input.rfind("/rag load ", 0) == 0) {
            auto& rag = RAGManager::getInstance();
            std::string path = input.substr(9);
            rag.loadDirectory(path);
            if (!rag.isEnabled()) {
                std::cout << "Documents loaded. RAG is currently \033[31mdisabled\033[0m." << std::endl;
                std::cout << "Use \033[32m/rag enable\033[0m to activate." << std::endl;
            }
            else {
                std::cout << "Documents loaded and RAG is enabled." << std::endl;
            }
            continue;
        }

        if (input.rfind("/rag load ", 0) == 0) {
            std::string path = input.substr(9);
            RAGManager::getInstance().loadDirectory(path);
            std::cout << "Loaded documents from: " << path << std::endl;
            continue;
        }

        if (input == "/rag clear") {
            RAGManager::getInstance().clear();
            std::cout << "Cleared all documents" << std::endl;
            continue;
        }

        if (input == "/rag list") {
            auto docs = RAGManager::getInstance().getAllDocuments();
            std::cout << "\n\033[33m=== Documents ===\033[0m" << std::endl;
            for (const auto& doc : docs) {
                std::cout << "[" << doc.id << "] " << doc.source << std::endl;
                std::cout << "    " << doc.content.substr(0, 100) << "..." << std::endl;
            }
            std::cout << "\033[33m==================\033[0m" << std::endl;
            continue;
        }

        // 新增命令：/cpu-moe <N>
        if (input.rfind("/cpu-moe ", 0) == 0) {
            std::string numStr = input.substr(9);
            try {
                int newCpuMoe = std::stoi(numStr);
                if (newCpuMoe < 0) newCpuMoe = 0;

                spdlog::info("Reloading model with CPU MoE layers = {}", newCpuMoe);

                // 关键：先彻底销毁旧引擎
                llm.reset();  // 调用析构函数，释放资源

                // 等待资源释放（尤其是 GPU 资源）
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                // 创建新引擎
                llm = std::make_unique<LLMEngine>(modelPath, newCpuMoe);

                if (!llm->isLoaded()) {
                    spdlog::error("Failed to reload model with new CPU MoE setting.");
                    // 可选：尝试恢复默认值或退出
                }
                else {
                    spdlog::info("Model reloaded successfully with CPU MoE layers = {}", newCpuMoe);
                }
            }
            catch (const std::exception& e) {
                spdlog::error("Invalid number: {}", numStr);
            }
            continue;
        }

        // 跳过空输入
        if (input.empty()) {
            continue;
        }

        // ========== 构建完整 Prompt（Qwen 格式）==========

        // 根据模式构建不同的系统提示
        auto currentMode = SkillManager::getInstance().getMode();
        std::string systemContent;

        if (currentMode == SkillMode::CHAT_ONLY) {
            systemContent =
                "You are CloseCrab, a helpful AI assistant.\n"
                "Answer questions directly. Do NOT use skills.\n"
                "Respond in the same language as the user.\n"
                "Be friendly and concise.";
        }
        else if (currentMode == SkillMode::SKILL_ONLY) {
            systemContent =
                "You are CloseCrab, a skill execution system.\n"
                "ONLY respond with SKILL format. No chatting.\n\n"
                "SKILL: skill_name\n"
                "PARAMS: param=value\n\n"
                "Available skills:\n"
                "- open_app: app_name, url (optional)\n"
                "- read_file: file_path\n"
                "- write_file: file_path, content\n"
                "- execute_command: command (dir, ls, echo, whoami, date, time, systeminfo, wmic)\n"
                "- search_web: query\n"
                "- system_info: type (cpu/memory/disk/all)\n"
                "- screenshot: file_path (optional)\n"
                "- clipboard: action (get/set), content (for set)\n"
                "- weather: city";
        }
        else {
            // AUTO 或 ASK 模式
            systemContent =
                "You are CloseCrab, an AI assistant.\n\n"
                "=== SKILLS ===\n"
                "When user asks you to DO something, use SKILL format:\n"
                "SKILL: skill_name\n"
                "PARAMS: param=value\n\n"
                "Available skills:\n"
                "- open_app: Open app or webpage. Params: app_name (notepad, calc, chrome, edge), url (optional)\n"
                "- read_file: Read file content. Params: file_path\n"
                "- write_file: Write content to file. Params: file_path, content\n"
                "- execute_command: Run system command. Params: command (dir, ls, echo, whoami, date, time, systeminfo, wmic)\n"
                "- search_web: Search the web. Params: query\n"
                "- system_info: Get system information. Params: type (cpu/memory/disk/all)\n"
                "- screenshot: Take screenshot. Params: file_path (optional, default desktop)\n"
                "- clipboard: Copy/paste text. Params: action (get/set), content (for set)\n"
                "- weather: Get weather. Params: city\n\n"
                "=== EXAMPLES ===\n"
                "User: 打开记事本\n"
                "You: SKILL: open_app\n"
                "PARAMS: app_name=notepad\n\n"
                "User: 用 Edge 打开百度\n"
                "You: SKILL: open_app\n"
                "PARAMS: app_name=edge, url=https://www.baidu.com\n\n"
                "User: 写文件 test.txt 内容 Hello\n"
                "You: SKILL: write_file\n"
                "PARAMS: file_path=test.txt, content=Hello\n\n"
                "User: 读取文件 test.txt\n"
                "You: SKILL: read_file\n"
                "PARAMS: file_path=test.txt\n\n"
                "User: 查看CPU信息\n"
                "You: SKILL: system_info\n"
                "PARAMS: type=cpu\n\n"
                "User: 截图\n"
                "You: SKILL: screenshot\n"
                "PARAMS: \n\n"
                "User: 复制 Hello 到剪贴板\n"
                "You: SKILL: clipboard\n"
                "PARAMS: action=set, content=Hello\n\n"
                "User: 剪贴板内容\n"
                "You: SKILL: clipboard\n"
                "PARAMS: action=get\n\n"
                "User: 北京天气\n"
                "You: SKILL: weather\n"
                "PARAMS: city=北京\n\n"
                "User: 现在几点\n"
                "You: SKILL: execute_command\n"
                "PARAMS: command=time /t\n\n"
                "User: 系统信息\n"
                "You: SKILL: execute_command\n"
                "PARAMS: command=systeminfo\n\n"
                "User: 你好\n"
                "You: 你好！有什么我可以帮你的？\n\n"
                "User: 1+1等于几\n"
                "You: 1+1等于2。\n\n"
                "=== RULES ===\n"
                "- DO something → use SKILL\n"
                "- ASK question → answer directly\n"
                "- Respond in same language as user\n"
                "- Be concise";
        }

        std::string fullPrompt = buildSystemMessage(systemContent);

        // ========== 新增：添加 RAG 上下文 ==========
        std::string ragContext = "";
        if (RAGManager::getInstance().isEnabled()) {
            ragContext = RAGManager::getInstance().buildRAGPrompt(input, 3);
        }
        if (!ragContext.empty()) {
            fullPrompt += "\n" + ragContext + "\n";
        }
        // ========== RAG 上下文添加结束 ==========

        // 2. 添加对话历史（只取最近 3 条，避免干扰）
        auto memories = memorySystem.getRecentMemories(sessionId, 20);  // 从 20 改为 3

        // 可选：添加一个简短的历史摘要而不是完整历史
        if (!memories.empty()) {
            fullPrompt += "Previous conversation to get referenced but remember you should be more concentrated on user's current asking:\n";
            for (const auto& mem : memories) {
                if (mem.role == "user") {
                    fullPrompt += "User: " + mem.content + "\n";
                }
                else {
                    fullPrompt += "Assistant: " + mem.content + "\n";
                }
            }
            fullPrompt += "\n";
        }

        // 3. 添加当前用户消息
        fullPrompt += buildUserMessage(input);

        // 4. 添加助手开始标记
        fullPrompt += "<|im_start|>assistant\n";

        // 5. 保存用户输入到数据库
        memorySystem.addMemory(sessionId, "user", input);

        // 调试信息（verbose 模式）
        if (verbose) {
            int totalTokens = llm->countTokens(fullPrompt);
            spdlog::debug("Full prompt tokens: {}", totalTokens);
            spdlog::debug("History entries: {}", memories.size());
        }

        // 显示思考提示
        std::cout << "\033[33mAI: \033[0m" << std::flush;

        std::string accumulatedResponse;
        bool firstToken = true;

        // 流式生成回复
        llm->generateRaw(
            fullPrompt,
            maxTokens,
            temperature,
            [&accumulatedResponse, &firstToken](const std::string& token) {
                if (firstToken) firstToken = false;
                std::string gbkToken = utf8ToGbk(token);
                std::cout << gbkToken << std::flush;
                accumulatedResponse += token;   // 仍然保留 UTF-8 用于存储
            },
            [&accumulatedResponse, &memorySystem, &sessionId]() {
                std::cout << std::endl;

                // ========== 处理 Skill 调用 ==========
                auto& skillManager = SkillManager::getInstance();
                std::string skillName;
                std::map<std::string, std::string> params;

                if (skillManager.shouldExecuteSkill(accumulatedResponse, skillName, params)) {
                    // 执行技能
                    std::string skillResult = skillManager.executeSkill(skillName, params);
                    std::cout << "\n[Skill Result]\n" << utf8ToGbk(skillResult) << std::endl;

                    // 保存技能结果到记忆
                    // 只存储技能结果，不加额外标记
                    //memorySystem.addMemory(sessionId, "assistant", skillResult);
                }
                else if (SkillManager::getInstance().getMode() != SkillMode::SKILL_ONLY) {
                    // 正常保存聊天回复
                    if (!accumulatedResponse.empty()) {
                        memorySystem.addMemory(sessionId, "assistant", accumulatedResponse);
                    }
                }
            }
        );
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::info("Session ID: {} (use -s {} to resume)", sessionId, sessionId);
    spdlog::info("CloseCrab finished!");
    // 清理
    if (httpServer) httpServer->stop();
    WSACleanup();
    return 0;
}