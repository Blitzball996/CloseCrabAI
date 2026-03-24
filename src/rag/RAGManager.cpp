#include "RAGManager.h"
#include "llm/LLMEngine.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <regex>
#include <sqlite3.h>
//#include "sqlite_vec.h"

RAGManager& RAGManager::getInstance() {
    static RAGManager instance;
    return instance;
}

RAGManager::~RAGManager() {
    if (db) {
        sqlite3_close(db);
        db = nullptr;
    }
}

bool RAGManager::init(const std::string& dbPath) {
    if (initialized) return true;

    // 打开数据库
    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to open vector database: {}", sqlite3_errmsg(db));
        return false;
    }

    /*
    // 加载 sqlite-vec 扩展
    rc = sqlite3_vec_init(db, nullptr, 0);
    if (rc != SQLITE_OK) {
        spdlog::warn("sqlite-vec extension not loaded, using fallback");
    }
    */

    // 创建表
    if (!createTables()) {
        return false;
    }

    initialized = true;
    spdlog::info("RAGManager initialized with database: {}", dbPath);

    // ========== 新增：初始化后保持禁用状态 ==========
    spdlog::info("RAG is disabled by default. Use /rag enable to activate");

    return true;
}

// ========== 新增：启用/禁用实现 ==========
void RAGManager::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_enabled = enabled;
    spdlog::info("RAG {}", enabled ? "enabled" : "disabled");
}

bool RAGManager::isEnabled() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_enabled;
}

void RAGManager::toggleEnabled() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_enabled = !m_enabled;
    spdlog::info("RAG {}", m_enabled ? "enabled" : "disabled");
}

bool RAGManager::createTables() {
    // 创建文档表
    const char* sql_docs = R"(
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    )";

    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql_docs, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        spdlog::error("Failed to create documents table: {}", errMsg);
        sqlite3_free(errMsg);
        return false;
    }

    // 创建向量表（使用 sqlite-vec）
    const char* sql_vectors = R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[768]
        );
    )";

    if (sqlite3_exec(db, sql_vectors, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        spdlog::warn("Failed to create vectors table: {} (using fallback)", errMsg);
        sqlite3_free(errMsg);
    }

    return true;
}

std::vector<float> RAGManager::embed(const std::string& text) {
    // 注意：需要 LLMEngine 支持 embedding
    // 这里先用简单方法：用 token 数量模拟（实际需要真正的 embedding）
    // 完整实现需要调用 llama.cpp 的 embedding API

    std::vector<float> result;
    result.resize(768, 0.0f);

    // 简单的哈希 embedding（仅用于演示）
    // 实际项目应该使用真正的 embedding 模型
    size_t hash = std::hash<std::string>{}(text);
    for (int i = 0; i < 768; ++i) {
        result[i] = (float)((hash >> (i % 64)) & 1) / 10.0f;
    }

    return result;
}

std::vector<std::string> RAGManager::splitText(const std::string& text, int chunkSize) {
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::string line;
    std::string current;

    while (std::getline(iss, line)) {
        if (current.length() + line.length() > chunkSize && !current.empty()) {
            chunks.push_back(current);
            current.clear();
        }
        current += line + "\n";
    }

    if (!current.empty()) {
        chunks.push_back(current);
    }

    return chunks;
}

// ========== 修改：检查启用状态 ==========
bool RAGManager::addDocument(const std::string& content, const std::string& source) {
    // 即使 RAG 禁用，也允许添加文档（方便用户提前加载）
    if (!initialized) {
        spdlog::error("RAGManager not initialized");
        return false;
    }

    // 分割长文档
    auto chunks = splitText(content, 500);

    for (const auto& chunk : chunks) {
        // 插入文档
        sqlite3_stmt* stmt;
        const char* sql = "INSERT INTO documents (content, source) VALUES (?, ?)";

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            spdlog::error("Failed to prepare insert: {}", sqlite3_errmsg(db));
            return false;
        }

        sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, source.c_str(), -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            spdlog::error("Failed to insert document: {}", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return false;
        }

        int docId = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(stmt);

        // 计算 embedding 并插入向量表
        std::vector<float> embedding = embed(chunk);

        // 构建向量插入语句
        std::string vectorSql = "INSERT INTO vectors (id, embedding) VALUES (?, ?)";
        sqlite3_stmt* vecStmt;

        if (sqlite3_prepare_v2(db, vectorSql.c_str(), -1, &vecStmt, nullptr) != SQLITE_OK) {
            spdlog::error("Failed to prepare vector insert: {}", sqlite3_errmsg(db));
            continue;
        }

        sqlite3_bind_int(vecStmt, 1, docId);
        sqlite3_bind_blob(vecStmt, 2, embedding.data(), embedding.size() * sizeof(float), SQLITE_STATIC);

        if (sqlite3_step(vecStmt) != SQLITE_DONE) {
            spdlog::error("Failed to insert vector: {}", sqlite3_errmsg(db));
        }

        sqlite3_finalize(vecStmt);
    }

    spdlog::info("Added document: {} ({} chunks)", source, chunks.size());
    return true;
}

bool RAGManager::addDocuments(const std::vector<std::pair<std::string, std::string>>& docs) {
    for (const auto& [content, source] : docs) {
        if (!addDocument(content, source)) {
            return false;
        }
    }
    return true;
}

// ========== 修改：检查启用状态 ==========
bool RAGManager::loadDirectory(const std::string& path) {
    namespace fs = std::filesystem;

    if (!fs::exists(path)) {
        spdlog::error("Directory not found: {}", path);
        return false;
    }

    // 即使 RAG 禁用，也允许加载文档
    int count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".md" || ext == ".cpp" || ext == ".h" || ext == ".py") {
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string content = buffer.str();
                    if (addDocument(content, entry.path().string())) {
                        count++;
                    }
                    file.close();
                }
            }
        }
    }

    spdlog::info("Loaded {} documents from {}", count, path);

    if (!isEnabled()) {
        spdlog::info("RAG is disabled, loaded documents will not be used until enabled");
    }

    return true;
}

// ========== 修改：检查启用状态 ==========
std::vector<Document> RAGManager::search(const std::string& query, int topK) {
    std::vector<Document> results;

    // ========== 新增：如果禁用，直接返回空结果 ==========
    if (!isEnabled()) {
        spdlog::debug("RAG is disabled, skipping search");
        return results;
    }

    if (!initialized) {
        spdlog::error("RAGManager not initialized");
        return results;
    }

    // 计算查询的 embedding
    std::vector<float> queryEmbed = embed(query);

    // 搜索相似向量
    std::string sql = R"(
        SELECT d.id, d.content, d.source, distance
        FROM vectors v
        JOIN documents d ON v.id = d.id
        WHERE v.embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare search: {}", sqlite3_errmsg(db));
        return results;
    }

    sqlite3_bind_blob(stmt, 1, queryEmbed.data(), queryEmbed.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, topK);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Document doc;
        doc.id = sqlite3_column_int(stmt, 0);
        doc.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        doc.source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        results.push_back(doc);
    }

    sqlite3_finalize(stmt);

    spdlog::debug("Found {} documents for query: {}", results.size(), query);
    return results;
}

// ========== 修改：检查启用状态 ==========
std::string RAGManager::buildRAGPrompt(const std::string& query, int topK) {
    // ========== 新增：如果禁用，直接返回空字符串 ==========
    if (!isEnabled()) {
        return "";
    }

    auto docs = search(query, topK);

    if (docs.empty()) {
        return "";
    }

    std::string prompt = "Here are some relevant documents for reference:\n\n";
    for (size_t i = 0; i < docs.size(); ++i) {
        prompt += "--- Document " + std::to_string(i + 1) + " ---\n";
        prompt += docs[i].content + "\n\n";
    }
    prompt += "Based on these documents, answer the following question:\n";
    prompt += "Question: " + query + "\n";
    prompt += "Answer: ";

    return prompt;
}

bool RAGManager::deleteDocument(int id) {
    if (!initialized) return false;

    // 删除向量
    const char* sqlVec = "DELETE FROM vectors WHERE id = ?";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sqlVec, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, id);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // 删除文档
    const char* sqlDoc = "DELETE FROM documents WHERE id = ?";
    if (sqlite3_prepare_v2(db, sqlDoc, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, id);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    return true;
}

void RAGManager::clear() {
    if (!initialized) return;

    sqlite3_exec(db, "DELETE FROM vectors", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "DELETE FROM documents", nullptr, nullptr, nullptr);
    spdlog::info("Cleared all documents");
}

int RAGManager::getDocumentCount() const {
    if (!initialized) return 0;

    sqlite3_stmt* stmt;
    int count = 0;

    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM documents", -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }

    return count;
}

std::vector<Document> RAGManager::getAllDocuments() const {
    std::vector<Document> results;

    if (!initialized) return results;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT id, content, source FROM documents ORDER BY id DESC";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Document doc;
            doc.id = sqlite3_column_int(stmt, 0);
            doc.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            doc.source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            results.push_back(doc);
        }
        sqlite3_finalize(stmt);
    }

    return results;
}