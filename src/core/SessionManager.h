#pragma once
#include <string>
#include <memory>
#include <sqlite3.h>
#include <filesystem>
#include <spdlog/spdlog.h>

struct Session {
    std::string id;
    std::string userId;
    std::string context;  // JSON 格式的对话上下文
    long long createdAt;
    long long updatedAt;
};

class SessionManager {
public:
    SessionManager(const std::string& dbPath);
    ~SessionManager();

    // 创建新会话
    std::string createSession(const std::string& userId);

    // 获取会话
    std::shared_ptr<Session> getSession(const std::string& sessionId);

    // 更新会话上下文
    bool updateContext(const std::string& sessionId, const std::string& context);

    // 删除会话
    bool deleteSession(const std::string& sessionId);

private:
    sqlite3* db;
    bool initDatabase();
};