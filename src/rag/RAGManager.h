#pragma once
#include <string>
#include <vector>
#include <memory>
#include <sqlite3.h>
#include <mutex>

struct Document {
    int id;
    std::string content;
    std::string source;
    std::vector<float> embedding;
};

class RAGManager {
public:
    static RAGManager& getInstance();

    // 初始化（创建数据库表）
    bool init(const std::string& dbPath = "data/vectors.db");

    // 添加文档
    bool addDocument(const std::string& content, const std::string& source = "");

    // 批量添加文档
    bool addDocuments(const std::vector<std::pair<std::string, std::string>>& docs);

    // 加载目录中的所有文本文件
    bool loadDirectory(const std::string& path);

    // 搜索相关文档
    std::vector<Document> search(const std::string& query, int topK = 3);

    // 构建 RAG prompt
    std::string buildRAGPrompt(const std::string& query, int topK = 3);

    // 删除文档
    bool deleteDocument(int id);

    // 清空所有文档
    void clear();

    // 获取文档数量
    int getDocumentCount() const;

    // 获取文档列表
    std::vector<Document> getAllDocuments() const;

    // ========== 新增：启用/禁用 RAG ==========
    void setEnabled(bool enabled);
    bool isEnabled() const;
    void toggleEnabled();  // 切换状态

private:
    RAGManager() = default;
    ~RAGManager();

    sqlite3* db = nullptr;
    bool initialized = false;

    // ========== 新增：启用标志和互斥锁 ==========
    bool m_enabled = false;  // 默认禁用
    mutable std::mutex m_mutex;  // 保护 m_enabled 的线程安全

    // 向量化文本（调用 LLMEngine 的 embedding）
    std::vector<float> embed(const std::string& text);

    // 创建向量表
    bool createTables();

    // 分割文本（用于长文档）
    std::vector<std::string> splitText(const std::string& text, int chunkSize = 500);
};