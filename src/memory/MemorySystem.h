#pragma once
#include <string>
#include <vector>
#include <sqlite3.h>

struct Memory {
    std::string id;
    std::string sessionId;
    std::string role;      // "user" 或 "assistant"
    std::string content;
    long long timestamp;
};

class MemorySystem {
public:
    MemorySystem(const std::string& dbPath);
    ~MemorySystem();

    // 添加记忆
    bool addMemory(const std::string& sessionId, const std::string& role, const std::string& content);

    // 获取会话的所有记忆
    std::vector<Memory> getMemories(const std::string& sessionId, int limit = 50);

    // 获取最近的记忆
    std::vector<Memory> getRecentMemories(const std::string& sessionId, int count = 10);

    // 清除会话记忆
    bool clearMemories(const std::string& sessionId);

private:
    sqlite3* db;
    bool initDatabase();
};