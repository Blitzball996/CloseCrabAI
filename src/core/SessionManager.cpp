#include "SessionManager.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <chrono>

SessionManager::SessionManager(const std::string& dbPath) : db(nullptr) {
    // 创建数据库文件所在的目录
    std::filesystem::path path(dbPath);
    auto parentPath = path.parent_path();
    if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
        std::filesystem::create_directories(parentPath);
        spdlog::info("Created directory: {}", parentPath.string());
    }

    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc) {
        spdlog::error("Can't open database: {}", sqlite3_errmsg(db));
        return;
    }
    initDatabase();
    spdlog::info("SessionManager initialized with database: {}", dbPath);
}

SessionManager::~SessionManager() {
    if (db) {
        sqlite3_close(db);
        spdlog::info("SessionManager closed");
    }
}

bool SessionManager::initDatabase() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            context TEXT,
            created_at INTEGER,
            updated_at INTEGER
        );
    )";

    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to create sessions table: {}", errMsg);
        sqlite3_free(errMsg);
        return false;
    }

    spdlog::info("Sessions table ready");
    return true;
}

std::string SessionManager::createSession(const std::string& userId) {
    // 生成简单的时间戳 ID
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string sessionId = userId + "_" + std::to_string(timestamp);

    const char* sql = "INSERT INTO sessions (id, user_id, context, created_at, updated_at) VALUES (?, ?, '{}', ?, ?)";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare insert: {}", sqlite3_errmsg(db));
        return "";
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, userId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 3, timestamp);
    sqlite3_bind_int64(stmt, 4, timestamp);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        spdlog::error("Failed to insert session: {}", sqlite3_errmsg(db));
        return "";
    }

    spdlog::info("Created session: {} for user: {}", sessionId, userId);
    return sessionId;
}

std::shared_ptr<Session> SessionManager::getSession(const std::string& sessionId) {
    const char* sql = "SELECT id, user_id, context, created_at, updated_at FROM sessions WHERE id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare select: {}", sqlite3_errmsg(db));
        return nullptr;
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        auto session = std::make_shared<Session>();
        session->id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        session->userId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        session->context = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        session->createdAt = sqlite3_column_int64(stmt, 3);
        session->updatedAt = sqlite3_column_int64(stmt, 4);
        sqlite3_finalize(stmt);
        return session;
    }

    sqlite3_finalize(stmt);
    return nullptr;
}

bool SessionManager::updateContext(const std::string& sessionId, const std::string& context) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    const char* sql = "UPDATE sessions SET context = ?, updated_at = ? WHERE id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare update: {}", sqlite3_errmsg(db));
        return false;
    }

    sqlite3_bind_text(stmt, 1, context.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, timestamp);
    sqlite3_bind_text(stmt, 3, sessionId.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

bool SessionManager::deleteSession(const std::string& sessionId) {
    const char* sql = "DELETE FROM sessions WHERE id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare delete: {}", sqlite3_errmsg(db));
        return false;
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    spdlog::info("Deleted session: {}", sessionId);
    return rc == SQLITE_DONE;
}