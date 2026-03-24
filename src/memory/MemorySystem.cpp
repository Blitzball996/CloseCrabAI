#include "MemorySystem.h"
#include <spdlog/spdlog.h>
#include <chrono>

MemorySystem::MemorySystem(const std::string& dbPath) : db(nullptr) {
    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc) {
        spdlog::error("Can't open database for memory: {}", sqlite3_errmsg(db));
        return;
    }
    initDatabase();
    spdlog::info("MemorySystem initialized");
}

MemorySystem::~MemorySystem() {
    if (db) {
        sqlite3_close(db);
    }
}

bool MemorySystem::initDatabase() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
    )";

    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to create memories table: {}", errMsg);
        sqlite3_free(errMsg);
        return false;
    }

    spdlog::info("Memories table ready");
    return true;
}

bool MemorySystem::addMemory(const std::string& sessionId, const std::string& role, const std::string& content) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    const char* sql = "INSERT INTO memories (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare insert memory: {}", sqlite3_errmsg(db));
        return false;
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, role.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, timestamp);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        spdlog::error("Failed to insert memory: {}", sqlite3_errmsg(db));
        return false;
    }

    return true;
}

std::vector<Memory> MemorySystem::getMemories(const std::string& sessionId, int limit) {
    std::vector<Memory> memories;

    const char* sql = "SELECT id, session_id, role, content, timestamp FROM memories WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare select memories: {}", sqlite3_errmsg(db));
        return memories;
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, limit);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory mem;
        mem.id = std::to_string(sqlite3_column_int64(stmt, 0));
        mem.sessionId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        mem.role = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        mem.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        mem.timestamp = sqlite3_column_int64(stmt, 4);
        memories.push_back(mem);
    }

    sqlite3_finalize(stmt);
    return memories;
}

std::vector<Memory> MemorySystem::getRecentMemories(const std::string& sessionId, int count) {
    return getMemories(sessionId, count);
}

bool MemorySystem::clearMemories(const std::string& sessionId) {
    const char* sql = "DELETE FROM memories WHERE session_id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare delete memories: {}", sqlite3_errmsg(db));
        return false;
    }

    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}