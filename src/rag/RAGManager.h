#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <map>
#include "EmbeddingEngine.h"
#include "RerankerEngine.h"

// FAISS 前向声明
namespace faiss {
    class Index;
}

// GPU 支持（条件编译）
#ifdef FAISS_GPU_ENABLED
namespace faiss {
    namespace gpu {
        class StandardGpuResources;
        class GpuIndexFlatIP;
        class GpuIndexIVFFlat;
    }
}
#endif

struct sqlite3;

struct Document {
    int id;
    std::string content;
    std::string source;
    std::vector<float> embedding;
};

enum class IndexType {
    FLAT,      // 精确搜索
    IVF,       // 倒排索引
    SQ         // 标量量化
};

enum class DeviceType {
    AUTO,      // 自动选择（优先 GPU）
    CPU,       // 强制 CPU
    GPU        // 强制 GPU
};

class RAGManager {
public:
    static RAGManager& getInstance();

    // 初始化 — 模型路径从 config.yaml 读取
    bool init(const std::string& dbPath = "data/vectors.db",
        IndexType type = IndexType::FLAT,
        DeviceType device = DeviceType::AUTO,
        int dimension = 768);

    // 添加文档
    bool addDocument(const std::string& content, const std::string& source = "");

    // 批量添加文档
    bool addDocuments(const std::vector<std::pair<std::string, std::string>>& docs);

    // 加载目录
    bool loadDirectory(const std::string& path);

    // 搜索
    std::vector<Document> search(const std::string& query, int topK = 5);

    // 构建 RAG prompt
    std::string buildRAGPrompt(const std::string& query, int topK = 5);

    // 删除文档
    bool deleteDocument(int id);

    // 清空所有文档
    void clear();

    // 获取文档数量
    int getDocumentCount() const;

    // 获取所有文档
    std::vector<Document> getAllDocuments() const;

    // 启用/禁用 RAG
    void setEnabled(bool enabled);
    bool isEnabled() const;
    void toggleEnabled();

    // 获取状态信息
    std::string getIndexInfo() const;
    std::string getDeviceStatus() const;

    // 手动切换设备
    bool switchDevice(DeviceType device);

    /// 检查 embedding / reranker 模型文件是否存在
    /// 返回缺失的模型名称列表（空 = 全部就绪）
    static std::vector<std::string> checkModelFiles();

    std::unique_ptr<EmbeddingEngine> embeddingEngine;
    std::unique_ptr<RerankerEngine> reranker;

private:
    RAGManager() = default;
    ~RAGManager();

    sqlite3* db = nullptr;
    bool initialized = false;
    bool m_enabled = false;
    mutable std::mutex m_mutex;

    // FAISS 索引
    faiss::Index* cpuIndex = nullptr;
    faiss::Index* gpuIndex = nullptr;
    faiss::Index* currentIndex = nullptr;
    DeviceType currentDevice = DeviceType::AUTO;

#ifdef FAISS_GPU_ENABLED
    std::shared_ptr<faiss::gpu::StandardGpuResources> gpuResources;
#endif

    int dimension = 768;
    IndexType indexType = IndexType::FLAT;

    // ID 映射: FAISS 索引位置 -> 数据库 ID
    std::vector<int64_t> idMap;
    std::map<int64_t, int> reverseIdMap;

    // 内部方法
    std::vector<float> embed(const std::string& text);
    std::vector<std::string> splitText(const std::string& text, int chunkSize = 500);

    bool createTables();
    bool loadIndexFromDatabase();
    void rebuildIndex();
    bool createCPUIndex();
    bool createGPUIndex();
    bool copyIndexToGPU();
    bool copyIndexToCPU();
    bool checkGPUAvailability();
    bool isGPUMemorySufficient();

    // 数据库操作
    bool insertDocumentToDB(int docId, const std::string& content,
        const std::string& source,
        const std::vector<float>& embedding);
    Document getDocumentFromDB(int docId);
    bool deleteDocumentFromDB(int docId);
};