#include "RAGManager.h"
#include "llm/LLMEngine.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <regex>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>  // 添加缺失的头文件

// FAISS 头文件
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>

#ifdef FAISS_GPU_ENABLED
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

extern "C" {
#include <sqlite3.h>
}

// ============================================
// 单例模式
// ============================================
RAGManager& RAGManager::getInstance() {
    static RAGManager instance;
    return instance;
}

RAGManager::~RAGManager() {
    if (cpuIndex) {
        delete cpuIndex;
        cpuIndex = nullptr;
    }
#ifdef FAISS_GPU_ENABLED
    if (gpuIndex) {
        delete gpuIndex;
        gpuIndex = nullptr;
    }
#endif
    if (db) {
        sqlite3_close(db);
        db = nullptr;
    }
}

// ============================================
// 启用/禁用
// ============================================
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

// ============================================
// GPU 可用性检查
// ============================================
#ifdef FAISS_GPU_ENABLED
bool RAGManager::checkGPUAvailability() {
    try {
        auto testResources = std::make_shared<faiss::gpu::StandardGpuResources>();
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexFlatIP testIndex(testResources.get(), 128, config);
        spdlog::info("GPU is available and working");
        return true;
    }
    catch (const std::exception& e) {
        spdlog::warn("GPU not available: {}", e.what());
        return false;
    }
}
#endif

// ============================================
// 创建 CPU 索引
// ============================================
bool RAGManager::createCPUIndex() {
    try {
        switch (indexType) {
        case IndexType::FLAT:
            cpuIndex = new faiss::IndexFlatIP(dimension);
            spdlog::info("Created CPU Flat index (dim={})", dimension);
            break;

        case IndexType::IVF: {
            auto* quantizer = new faiss::IndexFlatIP(dimension);
            int nlist = 100;
            cpuIndex = new faiss::IndexIVFFlat(quantizer, dimension, nlist, faiss::METRIC_INNER_PRODUCT);
            spdlog::info("Created CPU IVF index (nlist={})", nlist);
            break;
        }

        case IndexType::SQ:
            cpuIndex = new faiss::IndexScalarQuantizer(
                dimension,
                faiss::ScalarQuantizer::QT_8bit,
                faiss::METRIC_INNER_PRODUCT);
            spdlog::info("Created CPU SQ index");
            break;
        }
        return cpuIndex != nullptr;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to create CPU index: {}", e.what());
        return false;
    }
}

// ============================================
// 创建 GPU 索引
// ============================================
#ifdef FAISS_GPU_ENABLED
bool RAGManager::createGPUIndex() {
    try {
        if (!gpuResources) {
            gpuResources = std::make_shared<faiss::gpu::StandardGpuResources>();
            gpuResources->setTempMemory(512 * 1024 * 1024);
            gpuResources->setPinnedMemory(256 * 1024 * 1024);
            spdlog::info("GPU resources initialized");
        }

        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;

        switch (indexType) {
        case IndexType::FLAT: {
            gpuIndex = new faiss::gpu::GpuIndexFlatIP(
                gpuResources.get(),
                dimension,
                config
            );
            spdlog::info("Created GPU Flat index (dim={})", dimension);
            break;
        }

        case IndexType::IVF: {
            int nlist = 100;
            faiss::gpu::GpuIndexIVFFlatConfig ivfConfig;
            ivfConfig.device = 0;

            gpuIndex = new faiss::gpu::GpuIndexIVFFlat(
                gpuResources.get(),
                dimension,
                nlist,
                faiss::METRIC_INNER_PRODUCT,
                ivfConfig
            );
            spdlog::info("Created GPU IVF index (nlist={})", nlist);
            break;
        }

        default:
            spdlog::warn("GPU doesn't support {} index type", (int)indexType);
            return false;
        }
        return gpuIndex != nullptr;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to create GPU index: {}", e.what());
        return false;
    }
}
#endif

// ============================================
// 初始化
// ============================================
bool RAGManager::init(const std::string& dbPath, IndexType type,
    DeviceType device, int dim) {
    if (initialized) return true;

    dimension = dim;
    indexType = type;

    embeddingEngine = std::make_unique<EmbeddingEngine>(
        "models/model_quantized.onnx", true);

    reranker = std::make_unique<RerankerEngine>(
        "models/bge-reranker.onnx", true);

    // 创建数据库目录
    std::filesystem::path path(dbPath);
    std::filesystem::create_directories(path.parent_path());

    // 打开 SQLite 数据库
    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to open database: {}", sqlite3_errmsg(db));
        return false;
    }

    // 创建表
    if (!createTables()) {
        return false;
    }

    // 创建 CPU 索引（总是创建，作为 fallback）
    if (!createCPUIndex()) {
        spdlog::error("Failed to create CPU index");
        return false;
    }

    // 默认使用 CPU
    currentIndex = cpuIndex;
    currentDevice = DeviceType::CPU;

    // 尝试使用 GPU（如果启用且请求）
#ifdef FAISS_GPU_ENABLED
    if ((device == DeviceType::GPU || device == DeviceType::AUTO) && checkGPUAvailability()) {
        if (createGPUIndex()) {
            currentIndex = gpuIndex;
            currentDevice = DeviceType::GPU;
            spdlog::info("Using GPU device");
        }
        else {
            spdlog::warn("Failed to create GPU index, using CPU fallback");
        }
    }
#else
    if (device == DeviceType::GPU || device == DeviceType::AUTO) {
        spdlog::info("GPU support not compiled, using CPU");
    }
#endif

    // 从数据库加载现有数据
    if (!loadIndexFromDatabase()) {
        spdlog::info("No existing data found, starting with empty index");
    }

    initialized = true;
    spdlog::info("RAGManager initialized with {} device",
        currentDevice == DeviceType::GPU ? "GPU" : "CPU");
    spdlog::info("Document count: {}, Index size: {}",
        getDocumentCount(), currentIndex->ntotal);
    spdlog::info("RAG is disabled by default. Use /rag enable to activate");

    return true;
}
// ============================================
// 创建数据库表
// ============================================
bool RAGManager::createTables() {
    char* errMsg = nullptr;

    // 创建文档表
    const char* sql_docs = R"(
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    )";

    if (sqlite3_exec(db, sql_docs, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        spdlog::error("Failed to create documents table: {}", errMsg);
        sqlite3_free(errMsg);
        return false;
    }

    // 创建 FTS5 全文搜索表作为 fallback
    const char* sql_fts = R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content, 
            source,
            tokenize = 'porter unicode61'
        );
        
        CREATE TRIGGER IF NOT EXISTS sync_documents_fts_insert 
        AFTER INSERT ON documents
        BEGIN
            INSERT INTO documents_fts(rowid, content, source) 
            VALUES (new.id, new.content, new.source);
        END;
        
        CREATE TRIGGER IF NOT EXISTS sync_documents_fts_delete 
        AFTER DELETE ON documents
        BEGIN
            DELETE FROM documents_fts WHERE rowid = old.id;
        END;
    )";

    sqlite3_exec(db, sql_fts, nullptr, nullptr, nullptr);

    return true;
}

// ============================================
// 从数据库加载索引
// ============================================
bool RAGManager::loadIndexFromDatabase() {
    sqlite3_stmt* stmt;
    const char* sql = "SELECT id, embedding FROM documents WHERE embedding IS NOT NULL";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        spdlog::error("Failed to prepare load query: {}", sqlite3_errmsg(db));
        return false;
    }

    int count = 0;
    idMap.clear();
    reverseIdMap.clear();

    // 收集所有向量
    std::vector<float> allVectors;
    std::vector<int> docIds;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int docId = sqlite3_column_int(stmt, 0);
        const void* blob = sqlite3_column_blob(stmt, 1);
        int blobSize = sqlite3_column_bytes(stmt, 1);

        if (blob && blobSize == dimension * sizeof(float)) {
            std::vector<float> embedding(dimension);
            memcpy(embedding.data(), blob, blobSize);

            allVectors.insert(allVectors.end(), embedding.begin(), embedding.end());
            docIds.push_back(docId);
            count++;
        }
    }

    sqlite3_finalize(stmt);

    if (count > 0) {
        // 训练 IVF 索引（如果需要）
        if (indexType == IndexType::IVF) {
            auto* ivfIndex = dynamic_cast<faiss::IndexIVFFlat*>(cpuIndex);
            if (ivfIndex && !ivfIndex->is_trained) {
                spdlog::info("Training IVF index with {} vectors...", count);
                ivfIndex->train(count, allVectors.data());
            }
        }

        // 添加向量到 CPU 索引
        cpuIndex->add(count, allVectors.data());

        // 更新 ID 映射
        for (int i = 0; i < count; i++) {
            idMap.push_back(docIds[i]);
            reverseIdMap[docIds[i]] = i;
        }

        spdlog::info("Loaded {} vectors from database", count);

        // 如果当前使用 GPU，也添加到 GPU 索引
#ifdef FAISS_GPU_ENABLED
        if (currentDevice == DeviceType::GPU && gpuIndex) {
            // 对于 IVF，需要先训练
            if (indexType == IndexType::IVF) {
                auto* ivfGpuIndex = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(gpuIndex);
                if (ivfGpuIndex && !ivfGpuIndex->is_trained) {
                    spdlog::info("Training GPU IVF index...");
                    ivfGpuIndex->train(count, allVectors.data());
                }
            }
            spdlog::info("Adding {} vectors to GPU index...", count);
            gpuIndex->add(count, allVectors.data());
        }
#endif
    }

    return count > 0;
}

// ============================================
// 重建索引
// ============================================
void RAGManager::rebuildIndex() {
    std::lock_guard<std::mutex> lock(m_mutex);

    spdlog::info("Rebuilding index...");

    // 保存当前设备类型
    DeviceType targetDevice = currentDevice;

    // 删除旧索引
    if (cpuIndex) {
        delete cpuIndex;
        cpuIndex = nullptr;
    }
#ifdef FAISS_GPU_ENABLED
    if (gpuIndex) {
        delete gpuIndex;
        gpuIndex = nullptr;
    }
#endif

    // 重新创建 CPU 索引
    if (!createCPUIndex()) {
        spdlog::error("Failed to recreate CPU index");
        return;
    }

    // 根据目标设备设置当前索引
    if (targetDevice == DeviceType::GPU) {
#ifdef FAISS_GPU_ENABLED
        if (createGPUIndex()) {
            currentIndex = gpuIndex;
            currentDevice = DeviceType::GPU;
        }
        else {
            currentIndex = cpuIndex;
            currentDevice = DeviceType::CPU;
            spdlog::warn("Failed to recreate GPU index, using CPU");
        }
#else
        currentIndex = cpuIndex;
        currentDevice = DeviceType::CPU;
#endif
    }
    else {
        currentIndex = cpuIndex;
        currentDevice = DeviceType::CPU;
    }

    // 重新加载数据
    loadIndexFromDatabase();

    spdlog::info("Index rebuilt, new size: {}", currentIndex->ntotal);
}

// ============================================
// Embedding 函数（简化版，实际应使用真正的 embedding 模型）
// ============================================
std::vector<float> RAGManager::embed(const std::string& text) {
    return embeddingEngine->encode(text);
}

// ============================================
// 文本分块
// ============================================
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

// ============================================
// 添加文档
// ============================================
bool RAGManager::addDocument(const std::string& content, const std::string& source) {
    if (!initialized) {
        spdlog::error("RAGManager not initialized");
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    auto chunks = splitText(content, 500);

    for (const auto& chunk : chunks) {
        if (chunk.empty()) continue;

        // 生成 embedding
        std::vector<float> embedding = embed(chunk);

        // 插入到数据库
        const char* sql = "INSERT INTO documents (content, source, embedding) VALUES (?, ?, ?)";
        sqlite3_stmt* stmt;

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            spdlog::error("Failed to prepare insert: {}", sqlite3_errmsg(db));
            return false;
        }

        sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, source.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 3, embedding.data(),
            embedding.size() * sizeof(float), SQLITE_STATIC);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            spdlog::error("Failed to insert document: {}", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return false;
        }

        int docId = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(stmt);

        // 添加到 FAISS 索引
        currentIndex->add(1, embedding.data());

        // 更新 ID 映射
        int64_t faissId = idMap.size();
        idMap.push_back(docId);
        reverseIdMap[docId] = static_cast<int>(faissId);

        // 如果当前使用 GPU，同时添加到 GPU 索引
#ifdef FAISS_GPU_ENABLED
        if (currentDevice == DeviceType::GPU && gpuIndex && currentIndex != gpuIndex) {
            gpuIndex->add(1, embedding.data());
        }
#endif
    }

    spdlog::info("Added document: {} ({} chunks)", source, chunks.size());
    return true;
}

// ============================================
// 批量添加文档
// ============================================
bool RAGManager::addDocuments(const std::vector<std::pair<std::string, std::string>>& docs) {
    for (const auto& [content, source] : docs) {
        if (!addDocument(content, source)) {
            return false;
        }
    }
    return true;
}

// ============================================
// 加载目录
// ============================================
bool RAGManager::loadDirectory(const std::string& path) {
    namespace fs = std::filesystem;

    if (!fs::exists(path)) {
        spdlog::error("Directory not found: {}", path);
        return false;
    }

    int count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".md" || ext == ".cpp" ||
                ext == ".h" || ext == ".py" || ext == ".json") {
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
    return true;
}

// ============================================
// 搜索
// ============================================
std::vector<Document> RAGManager::search(const std::string& query, int topK) {

    std::vector<Document> results;

    if (!isEnabled()) return results;

    auto queryEmbed = embeddingEngine->encode(query);

    std::vector<float> distances(topK * 5);
    std::vector<faiss::idx_t> indices(topK * 5);

    //先多召回（重要）
    currentIndex->search(1, queryEmbed.data(), topK * 5,
        distances.data(), indices.data());

    struct Candidate {
        Document doc;
        float score;
    };

    std::vector<Candidate> candidates;

    for (int i = 0; i < topK * 5; i++) {
        if (indices[i] < 0) continue;

        int docId = idMap[indices[i]];
        auto doc = getDocumentFromDB(docId);

        float score = reranker->score(query, doc.content);

        candidates.push_back({ doc, score });
    }

    // rerank排序
    std::sort(candidates.begin(), candidates.end(),
        [](auto& a, auto& b) {
            return a.score > b.score;
        });

    for (int i = 0; i < topK && i < candidates.size(); i++) {
        results.push_back(candidates[i].doc);
    }

    return results;
}

// ============================================
// 构建 RAG Prompt
// ============================================
std::string RAGManager::buildRAGPrompt(const std::string& query, int topK) {
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

// ============================================
// 从数据库获取文档
// ============================================
Document RAGManager::getDocumentFromDB(int docId) {
    Document doc;
    doc.id = -1;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT id, content, source FROM documents WHERE id = ?";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, docId);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            doc.id = sqlite3_column_int(stmt, 0);
            const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

            if (content) doc.content = content;
            if (source) doc.source = source;
        }

        sqlite3_finalize(stmt);
    }

    return doc;
}

// ============================================
// 删除文档
// ============================================
bool RAGManager::deleteDocumentFromDB(int docId) {
    const char* sql = "DELETE FROM documents WHERE id = ?";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, docId);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        // 重建索引以移除已删除的文档
        rebuildIndex();
        return true;
    }

    return false;
}

bool RAGManager::deleteDocument(int id) {
    if (!initialized) return false;

    std::lock_guard<std::mutex> lock(m_mutex);

    // 从映射中删除
    auto it = reverseIdMap.find(id);
    if (it != reverseIdMap.end()) {
        reverseIdMap.erase(it);
    }

    // 从数据库删除
    return deleteDocumentFromDB(id);
}

// ============================================
// 清空所有文档
// ============================================
void RAGManager::clear() {
    if (!initialized) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    // 清空数据库
    sqlite3_exec(db, "DELETE FROM documents", nullptr, nullptr, nullptr);

    // 重建索引
    rebuildIndex();

    spdlog::info("Cleared all documents");
}

// ============================================
// 获取文档数量
// ============================================
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

// ============================================
// 获取所有文档
// ============================================
std::vector<Document> RAGManager::getAllDocuments() const {
    std::vector<Document> results;

    if (!initialized) return results;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT id, content, source FROM documents ORDER BY id DESC";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Document doc;
            doc.id = sqlite3_column_int(stmt, 0);
            const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

            if (content) doc.content = content;
            if (source) doc.source = source;

            results.push_back(doc);
        }
        sqlite3_finalize(stmt);
    }

    return results;
}

// ============================================
// 切换设备
// ============================================
bool RAGManager::switchDevice(DeviceType device) {
    if (!initialized) return false;

    std::lock_guard<std::mutex> lock(m_mutex);

    if (device == currentDevice) {
        return true;
    }

#ifdef FAISS_GPU_ENABLED
    if (device == DeviceType::GPU) {
        if (checkGPUAvailability() && createGPUIndex()) {
            // 复制数据到 GPU
            if (cpuIndex->ntotal > 0) {
                spdlog::info("Copying {} vectors to GPU...", cpuIndex->ntotal);
                // 重新加载数据到 GPU
                loadIndexFromDatabase();
            }
            currentIndex = gpuIndex;
            currentDevice = DeviceType::GPU;
            spdlog::info("Switched to GPU device");
            return true;
        }
        else {
            spdlog::warn("Failed to switch to GPU");
            return false;
        }
    }
#endif

    if (device == DeviceType::CPU) {
        currentIndex = cpuIndex;
        currentDevice = DeviceType::CPU;
        spdlog::info("Switched to CPU device");
        return true;
    }

    return false;
}

// ============================================
// 获取索引信息
// ============================================
std::string RAGManager::getIndexInfo() const {
    if (!initialized || !currentIndex) return "Not initialized";

    std::string info = "FAISS Index Type: ";
    switch (indexType) {
    case IndexType::FLAT: info += "Flat"; break;
    case IndexType::IVF: info += "IVF"; break;
    case IndexType::SQ: info += "ScalarQuantizer"; break;
    }

    info += "\nCurrent device: ";
    info += (currentDevice == DeviceType::GPU) ? "GPU" : "CPU";

    info += "\nTotal vectors: " + std::to_string(currentIndex->ntotal);
    info += "\nDimension: " + std::to_string(dimension);
    info += "\nDocuments in DB: " + std::to_string(getDocumentCount());

    return info;
}

std::string RAGManager::getDeviceStatus() const {
    std::string status = "Current device: ";
    status += (currentDevice == DeviceType::GPU) ? "GPU" : "CPU";

    status += "\nGPU available: ";
#ifdef FAISS_GPU_ENABLED
    // 注意：这里不调用 checkGPUAvailability()，因为它是非 const 的
    // 直接返回编译时信息
    status += "Yes (if CUDA is properly installed)";
#else
    status += "No (compiled without GPU support)";
#endif

    if (currentIndex) {
        status += "\nIndex size: " + std::to_string(currentIndex->ntotal);
    }
    status += "\nDimension: " + std::to_string(dimension);

    return status;
}