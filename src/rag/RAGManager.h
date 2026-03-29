#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <map>
#include "EmbeddingEngine.h"
#include "RerankerEngine.h"

namespace faiss { class Index; }
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

enum class IndexType { FLAT, IVF, SQ };
enum class DeviceType { AUTO, CPU, GPU };

class RAGManager {
public:
    static RAGManager& getInstance();

    bool init(const std::string& dbPath = "data/vectors.db",
        IndexType type = IndexType::FLAT,
        DeviceType device = DeviceType::AUTO,
        int dimension = 768);

    bool addDocument(const std::string& content, const std::string& source = "");
    bool addDocuments(const std::vector<std::pair<std::string, std::string>>& docs);
    bool loadDirectory(const std::string& path);
    std::vector<Document> search(const std::string& query, int topK = 5);
    std::string buildRAGPrompt(const std::string& query, int topK = 5);
    bool deleteDocument(int id);
    void clear();
    int getDocumentCount() const;
    std::vector<Document> getAllDocuments() const;
    void setEnabled(bool enabled);
    bool isEnabled() const;
    void toggleEnabled();
    std::string getIndexInfo() const;
    std::string getDeviceStatus() const;
    bool switchDevice(DeviceType device);

    /// 检查模型文件是否存在，返回缺失列表
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

    faiss::Index* cpuIndex = nullptr;
    faiss::Index* gpuIndex = nullptr;
    faiss::Index* currentIndex = nullptr;
    DeviceType currentDevice = DeviceType::AUTO;
#ifdef FAISS_GPU_ENABLED
    std::shared_ptr<faiss::gpu::StandardGpuResources> gpuResources;
#endif

    int dimension = 768;
    IndexType indexType = IndexType::FLAT;
    std::vector<int64_t> idMap;
    std::map<int64_t, int> reverseIdMap;

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
    bool insertDocumentToDB(int docId, const std::string& content,
        const std::string& source, const std::vector<float>& embedding);
    Document getDocumentFromDB(int docId);
    bool deleteDocumentFromDB(int docId);
};