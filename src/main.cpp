#include <iostream>
#include <thread>
#include <chrono>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include "config/Config.h"
#include "core/SessionManager.h"
#include "memory/MemorySystem.h"
#include "core/ThreadPool.h"
#include "llm/LLMEngine.h"

int main(int argc, char* argv[])
{
    // ============================================
    // 1. 解析命令行参数
    // ============================================
    CLI::App app{ "CloseCrab - AI Execution Engine" };

    std::string configFile = "config/config.yaml";
    bool verbose = false;

    app.add_option("-c,--config", configFile, "Config file path");
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    try {
        CLI11_PARSE(app, argc, argv);
    }
    catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // ============================================
    // 2. 加载配置
    // ============================================
    auto& config = Config::getInstance();
    if (!config.load(configFile)) {
        spdlog::error("Failed to load config, exiting...");
        return 1;
    }

    // ============================================
    // 3. 设置日志级别
    // ============================================
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

    spdlog::info("========================================");
    spdlog::info("CloseCrab v1.0.0 starting...");
    spdlog::info("========================================");

    // 获取配置参数
    int serverPort = config.getInt("server.port", 9001);
    std::string serverHost = config.getString("server.host", "127.0.0.1");
    std::string dbPath = config.getString("database.path", "data/closecrab.db");

    spdlog::info("Server: {}:{}", serverHost, serverPort);
    spdlog::info("Database: {}", dbPath);

    // ============================================
    // 4. 测试 SessionManager
    // ============================================
    spdlog::info("--- Testing SessionManager ---");
    SessionManager sessionManager(dbPath);

    std::string sessionId = sessionManager.createSession("test_user");
    if (sessionId.empty()) {
        spdlog::error("Failed to create session");
        return 1;
    }
    spdlog::info("Created session: {}", sessionId);

    auto session = sessionManager.getSession(sessionId);
    if (session) {
        spdlog::info("Retrieved session: {} for user: {}", session->id, session->userId);
    }

    // ============================================
    // 5. 测试 MemorySystem
    // ============================================
    spdlog::info("--- Testing MemorySystem ---");
    MemorySystem memorySystem(dbPath);

    // 添加记忆
    memorySystem.addMemory(sessionId, "user", "Hello, how are you?");
    memorySystem.addMemory(sessionId, "assistant", "I'm doing great! How can I help you?");
    memorySystem.addMemory(sessionId, "user", "Tell me about CloseCrab");
    memorySystem.addMemory(sessionId, "assistant", "CloseCrab is an AI execution engine written in C++");

    // 获取记忆
    auto memories = memorySystem.getRecentMemories(sessionId, 10);
    spdlog::info("Found {} memories for session", memories.size());
    for (const auto& mem : memories) {
        spdlog::info("  [{}] {}", mem.role, mem.content);
    }

    // ============================================
    // 6. 测试 ThreadPool
    // ============================================
    auto& threadPool = ThreadPool::getInstance();
    spdlog::info("Thread pool size: {}", threadPool.getThreadCount());

    // 测试1：有返回值的任务 - 使用 lambda 包装
    auto future1 = threadPool.submit([]() -> int {
        spdlog::info("Task 1: Calculating...");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return 42;
        });

    // 测试2：无返回值的任务
    auto future2 = threadPool.submit([]() {
        spdlog::info("Task 2: Processing...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        spdlog::info("Task 2: Done!");
        });

    // 测试3：带参数的任务 - 用 lambda 捕获参数
    int a = 10, b = 20;
    auto future3 = threadPool.submit([a, b]() -> int {
        spdlog::info("Task 3: Computing {}+{}...", a, b);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return a + b;
        });

    // 等待并获取结果
    int result1 = future1.get();
    spdlog::info("Task 1 result: {}", result1);

    future2.get();
    spdlog::info("Task 2 completed");

    int result3 = future3.get();
    spdlog::info("Task 3 result: {} + {} = {}", a, b, result3);

    // 测试4：批量任务
    spdlog::info("Testing batch tasks...");
    std::vector<std::future<int>> batchResults;
    for (int i = 0; i < 5; ++i) {
        batchResults.push_back(threadPool.submit([i]() -> int {
            spdlog::info("Batch task {} running", i);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i * i;
            }));
    }

    for (size_t i = 0; i < batchResults.size(); ++i) {
        int res = batchResults[i].get();
        spdlog::info("Batch task {} result: {}", i, res);
    }

    // ============================================
    // 8. 测试 LLMEngine
    // ============================================
    spdlog::info("--- Testing LLMEngine ---");

    std::string modelPath = config.getString("llm.model_path", "models/qwen2.5-14b-instruct-uncensored-q4_k_m.gguf");
    LLMEngine llm(modelPath);

    if (llm.isLoaded()) {
        spdlog::info("Model loaded: {}", llm.getModelInfo());

        std::string prompt = "What is C++? Answer in one sentence.";
        spdlog::info("Prompt: {}", prompt);

        std::string response = llm.generate(prompt, "You are a helpful assistant.", 256, 0.7f);
        spdlog::info("Response: {}", response);
    }
    else {
        spdlog::error("Failed to load model, skipping LLM test");
    }

    // ============================================
    // 9. 完成
    // ============================================
    spdlog::info("========================================");
    spdlog::info("CloseCrab completed successfully!");
    spdlog::info("========================================");

    return 0;  // ← 把 return 0 移到这里
}