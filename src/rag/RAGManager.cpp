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
#include <unordered_map>

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

#include <yaml-cpp/yaml.h>

// ============================================
// config.yaml → RAG 配置
// ============================================
namespace {
    struct RAGConfig {
        // Embedding (onnx-community/bge-small-zh-v1.5-ONNX)
        std::string embeddingModelPath = "models/bge-small-zh/onnx/model_quantized.onnx";
        std::string embeddingTokenizerPath = "models/bge-small-zh/tokenizer.json";
        // Reranker (onnx-community/bge-reranker-base-ONNX 或类似)
        std::string rerankerModelPath = "models/bge-reranker-base/onnx/model_uint8.onnx";
        std::string rerankerTokenizerPath = "models/bge-reranker-base/tokenizer.json";
    };

    RAGConfig loadRAGConfig(const std::string& configFile = "config/config.yaml") {
        RAGConfig cfg;
        try {
            if (!std::filesystem::exists(configFile)) {
                spdlog::warn("Config file not found: {}, using defaults", configFile);
                return cfg;
            }
            YAML::Node root = YAML::LoadFile(configFile);
            if (root["rag"]) {
                auto rag = root["rag"];
                if (rag["embedding_model_path"])
                    cfg.embeddingModelPath = rag["embedding_model_path"].as<std::string>();
                if (rag["embedding_tokenizer_path"])
                    cfg.embeddingTokenizerPath = rag["embedding_tokenizer_path"].as<std::string>();
                if (rag["reranker_model_path"])
                    cfg.rerankerModelPath = rag["reranker_model_path"].as<std::string>();
                if (rag["reranker_tokenizer_path"])
                    cfg.rerankerTokenizerPath = rag["reranker_tokenizer_path"].as<std::string>();
            }
        }
        catch (const std::exception& e) {
            spdlog::warn("Failed to parse config: {}, using defaults", e.what());
        }
        return cfg;
    }
}

// ============================================
// 单例
// ============================================
RAGManager& RAGManager::getInstance() {
    static RAGManager instance;
    return instance;
}

RAGManager::~RAGManager() {
    if (cpuIndex) { delete cpuIndex; cpuIndex = nullptr; }
#ifdef FAISS_GPU_ENABLED
    if (gpuIndex) { delete gpuIndex; gpuIndex = nullptr; }
#endif
    if (db) { sqlite3_close(db); db = nullptr; }
}

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
// 检查模型文件
// ============================================
std::vector<std::string> RAGManager::checkModelFiles() {
    auto cfg = loadRAGConfig();
    std::vector<std::string> missing;

    if (!std::filesystem::exists(cfg.embeddingModelPath))
        missing.push_back("Embedding ONNX: " + cfg.embeddingModelPath);
    if (!std::filesystem::exists(cfg.embeddingTokenizerPath))
        missing.push_back("Embedding tokenizer.json: " + cfg.embeddingTokenizerPath);
    if (!std::filesystem::exists(cfg.rerankerModelPath))
        missing.push_back("Reranker ONNX: " + cfg.rerankerModelPath);
    if (!std::filesystem::exists(cfg.rerankerTokenizerPath))
        missing.push_back("Reranker tokenizer.json: " + cfg.rerankerTokenizerPath);

    // 检查 model.onnx_data（如存在则需要同目录）
    auto checkOnnxData = [&](const std::string& onnxPath) {
        std::filesystem::path p(onnxPath);
        auto dataPath = p.parent_path() / (p.filename().string() + "_data");
        // 不报 missing，只做 info（model.onnx_data 不是所有模型都有）
        if (std::filesystem::exists(onnxPath) && std::filesystem::exists(dataPath)) {
            spdlog::info("Found external data: {}", dataPath.string());
        }
        };
    checkOnnxData(cfg.embeddingModelPath);
    checkOnnxData(cfg.rerankerModelPath);

    if (!missing.empty()) {
        spdlog::warn("Missing RAG model files:");
        for (const auto& m : missing)
            spdlog::warn("  - {}", m);
        spdlog::warn("Run download_model.bat to download required models.");
    }
    return missing;
}

// ============================================
// GPU 检测
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

bool RAGManager::createCPUIndex() {
    try {
        switch (indexType) {
        case IndexType::FLAT:
            cpuIndex = new faiss::IndexFlatIP(dimension);
            spdlog::info("Created CPU Flat index (dim={})", dimension);
            break;
        case IndexType::IVF: {
            auto* quantizer = new faiss::IndexFlatIP(dimension);
            cpuIndex = new faiss::IndexIVFFlat(quantizer, dimension, 100, faiss::METRIC_INNER_PRODUCT);
            spdlog::info("Created CPU IVF index");
            break;
        }
        case IndexType::SQ:
            cpuIndex = new faiss::IndexScalarQuantizer(
                dimension, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_INNER_PRODUCT);
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

#ifdef FAISS_GPU_ENABLED
bool RAGManager::createGPUIndex() {
    try {
        if (!gpuResources) {
            gpuResources = std::make_shared<faiss::gpu::StandardGpuResources>();
            gpuResources->setTempMemory(512 * 1024 * 1024);
            gpuResources->setPinnedMemory(256 * 1024 * 1024);
        }
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        switch (indexType) {
        case IndexType::FLAT:
            gpuIndex = new faiss::gpu::GpuIndexFlatIP(gpuResources.get(), dimension, config);
            break;
        case IndexType::IVF: {
            faiss::gpu::GpuIndexIVFFlatConfig ivfConfig;
            ivfConfig.device = 0;
            gpuIndex = new faiss::gpu::GpuIndexIVFFlat(
                gpuResources.get(), dimension, 100, faiss::METRIC_INNER_PRODUCT, ivfConfig);
            break;
        }
        default:
            spdlog::warn("GPU doesn't support this index type");
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
// 初始化 — 从 config.yaml 读取模型路径
// ============================================
bool RAGManager::init(const std::string& dbPath, IndexType type,
    DeviceType device, int dim)
{
    if (initialized) return true;
    dimension = dim;
    indexType = type;

    // 从 config.yaml 读取
    auto cfg = loadRAGConfig();
    auto missing = checkModelFiles();

    // 加载 Embedding (model.onnx + tokenizer.json)
    try {
        embeddingEngine = std::make_unique<EmbeddingEngine>(
            cfg.embeddingModelPath, cfg.embeddingTokenizerPath, true);
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to load embedding: {}", e.what());
    }

    // 加载 Reranker (model.onnx + tokenizer.json)
    try {
        reranker = std::make_unique<RerankerEngine>(
            cfg.rerankerModelPath, cfg.rerankerTokenizerPath, true);
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to load reranker: {}", e.what());
    }

    // 数据库
    std::filesystem::path path(dbPath);
    std::filesystem::create_directories(path.parent_path());
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        spdlog::error("Failed to open database: {}", sqlite3_errmsg(db));
        return false;
    }
    if (!createTables()) return false;

    // CPU 索引
    if (!createCPUIndex()) {
        spdlog::error("Failed to create CPU index");
        return false;
    }
    currentIndex = cpuIndex;
    currentDevice = DeviceType::CPU;

#ifdef FAISS_GPU_ENABLED
    if ((device == DeviceType::GPU || device == DeviceType::AUTO) && checkGPUAvailability()) {
        if (createGPUIndex()) {
            currentIndex = gpuIndex;
            currentDevice = DeviceType::GPU;
            spdlog::info("Using GPU device");
        }
    }
#else
    if (device == DeviceType::GPU || device == DeviceType::AUTO)
        spdlog::info("GPU support not compiled, using CPU");
#endif

    if (!loadIndexFromDatabase())
        spdlog::info("No existing data found, starting with empty index");

    initialized = true;
    spdlog::info("RAGManager initialized ({} device, {} docs, {} vectors)",
        currentDevice == DeviceType::GPU ? "GPU" : "CPU",
        getDocumentCount(), currentIndex->ntotal);
    return true;
}

// ============================================
// 以下方法和原始代码完全相同（省略注释中的乱码）
// ============================================
bool RAGManager::createTables() {
    char* errMsg = nullptr;
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
    const char* sql_fts = R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content, source, tokenize = 'porter unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS sync_documents_fts_insert
        AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, content, source)
            VALUES (new.id, new.content, new.source);
        END;
        CREATE TRIGGER IF NOT EXISTS sync_documents_fts_delete
        AFTER DELETE ON documents BEGIN
            DELETE FROM documents_fts WHERE rowid = old.id;
        END;
    )";
    sqlite3_exec(db, sql_fts, nullptr, nullptr, nullptr);
    return true;
}

bool RAGManager::loadIndexFromDatabase() {
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "SELECT id, embedding FROM documents WHERE embedding IS NOT NULL",
        -1, &stmt, nullptr) != SQLITE_OK) return false;
    int count = 0;
    idMap.clear(); reverseIdMap.clear();
    std::vector<float> allVectors;
    std::vector<int> docIds;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int docId = sqlite3_column_int(stmt, 0);
        const void* blob = sqlite3_column_blob(stmt, 1);
        int blobSize = sqlite3_column_bytes(stmt, 1);
        if (blob && blobSize == dimension * (int)sizeof(float)) {
            std::vector<float> emb(dimension);
            memcpy(emb.data(), blob, blobSize);
            allVectors.insert(allVectors.end(), emb.begin(), emb.end());
            docIds.push_back(docId);
            count++;
        }
    }
    sqlite3_finalize(stmt);
    if (count > 0) {
        if (indexType == IndexType::IVF) {
            auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(cpuIndex);
            if (ivf && !ivf->is_trained) ivf->train(count, allVectors.data());
        }
        cpuIndex->add(count, allVectors.data());
        for (int i = 0; i < count; i++) {
            idMap.push_back(docIds[i]);
            reverseIdMap[docIds[i]] = i;
        }
#ifdef FAISS_GPU_ENABLED
        if (currentDevice == DeviceType::GPU && gpuIndex) {
            if (indexType == IndexType::IVF) {
                auto* ivfGpu = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(gpuIndex);
                if (ivfGpu && !ivfGpu->is_trained) ivfGpu->train(count, allVectors.data());
            }
            gpuIndex->add(count, allVectors.data());
        }
#endif
        spdlog::info("Loaded {} vectors from database", count);
    }
    return count > 0;
}

void RAGManager::rebuildIndex() {
    std::lock_guard<std::mutex> lock(m_mutex);
    DeviceType target = currentDevice;
    if (cpuIndex) { delete cpuIndex; cpuIndex = nullptr; }
#ifdef FAISS_GPU_ENABLED
    if (gpuIndex) { delete gpuIndex; gpuIndex = nullptr; }
#endif
    createCPUIndex();
    if (target == DeviceType::GPU) {
#ifdef FAISS_GPU_ENABLED
        if (createGPUIndex()) { currentIndex = gpuIndex; currentDevice = DeviceType::GPU; }
        else
#endif
        {
            currentIndex = cpuIndex; currentDevice = DeviceType::CPU;
        }
    }
    else {
        currentIndex = cpuIndex; currentDevice = DeviceType::CPU;
    }
    loadIndexFromDatabase();
}

std::vector<float> RAGManager::embed(const std::string& text) {
    if (!embeddingEngine) return std::vector<float>(dimension, 0.0f);
    return embeddingEngine->encode(text);
}

std::vector<std::string> RAGManager::splitText(const std::string& text, int chunkSize) {
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::string line, current;
    while (std::getline(iss, line)) {
        if (current.length() + line.length() > (size_t)chunkSize && !current.empty()) {
            chunks.push_back(current); current.clear();
        }
        current += line + "\n";
    }
    if (!current.empty()) chunks.push_back(current);
    return chunks;
}

bool RAGManager::addDocument(const std::string& content, const std::string& source) {
    if (!initialized || !embeddingEngine) return false;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& chunk : splitText(content, 500)) {
        if (chunk.empty()) continue;
        auto embedding = embed(chunk);
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "INSERT INTO documents (content, source, embedding) VALUES (?, ?, ?)",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, source.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 3, embedding.data(), embedding.size() * sizeof(float), SQLITE_STATIC);
        if (sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return false; }
        int docId = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(stmt);
        currentIndex->add(1, embedding.data());
        idMap.push_back(docId);
        reverseIdMap[docId] = static_cast<int>(idMap.size() - 1);
    }
    return true;
}

bool RAGManager::addDocuments(const std::vector<std::pair<std::string, std::string>>& docs) {
    for (const auto& [c, s] : docs) if (!addDocument(c, s)) return false;
    return true;
}

bool RAGManager::loadDirectory(const std::string& path) {
    namespace fs = std::filesystem;
    if (!fs::exists(path)) return false;
    int count = 0;
    for (const auto& e : fs::recursive_directory_iterator(path)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".txt" || ext == ".md" || ext == ".cpp" ||
            ext == ".h" || ext == ".py" || ext == ".json") {
            std::ifstream f(e.path());
            std::stringstream buf; buf << f.rdbuf();
            if (addDocument(buf.str(), e.path().string())) count++;
        }
    }
    spdlog::info("Loaded {} documents from {}", count, path);
    return true;
}

std::vector<Document> RAGManager::search(const std::string& query, int topK) {
    std::vector<Document> results;
    if (!isEnabled() || !embeddingEngine) return results;
    auto qEmb = embeddingEngine->encode(query);
    int fetchK = topK * 5;
    std::vector<float> distances(fetchK);
    std::vector<faiss::idx_t> indices(fetchK);
    currentIndex->search(1, qEmb.data(), fetchK, distances.data(), indices.data());

    struct Candidate { Document doc; float score; };
    std::vector<Candidate> candidates;
    for (int i = 0; i < fetchK; i++) {
        if (indices[i] < 0) continue;
        auto doc = getDocumentFromDB(idMap[indices[i]]);
        float s = reranker ? reranker->score(query, doc.content) : distances[i];
        candidates.push_back({ doc, s });
    }
    std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.score > b.score; });
    for (int i = 0; i < topK && i < (int)candidates.size(); i++)
        results.push_back(candidates[i].doc);
    return results;
}

std::string RAGManager::buildRAGPrompt(const std::string& query, int topK) {
    if (!isEnabled()) return "";
    auto docs = search(query, topK);
    if (docs.empty()) return "";
    std::string prompt = "Here are some relevant documents for reference:\n\n";
    for (size_t i = 0; i < docs.size(); ++i)
        prompt += "--- Document " + std::to_string(i + 1) + " ---\n" + docs[i].content + "\n\n";
    prompt += "Based on these documents, answer the following question:\nQuestion: " + query + "\nAnswer: ";
    return prompt;
}

Document RAGManager::getDocumentFromDB(int docId) {
    Document doc; doc.id = -1;
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "SELECT id, content, source FROM documents WHERE id = ?",
        -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, docId);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            doc.id = sqlite3_column_int(stmt, 0);
            auto* c = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            auto* s = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            if (c) doc.content = c;
            if (s) doc.source = s;
        }
        sqlite3_finalize(stmt);
    }
    return doc;
}

bool RAGManager::deleteDocumentFromDB(int docId) {
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "DELETE FROM documents WHERE id = ?", -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, docId);
        sqlite3_step(stmt); sqlite3_finalize(stmt);
        rebuildIndex();
        return true;
    }
    return false;
}

bool RAGManager::deleteDocument(int id) {
    if (!initialized) return false;
    std::lock_guard<std::mutex> lock(m_mutex);
    reverseIdMap.erase(id);
    return deleteDocumentFromDB(id);
}

void RAGManager::clear() {
    if (!initialized) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    sqlite3_exec(db, "DELETE FROM documents", nullptr, nullptr, nullptr);
    rebuildIndex();
}

int RAGManager::getDocumentCount() const {
    if (!initialized) return 0;
    sqlite3_stmt* stmt; int count = 0;
    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM documents", -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) count = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
    }
    return count;
}

std::vector<Document> RAGManager::getAllDocuments() const {
    std::vector<Document> results;
    if (!initialized) return results;
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "SELECT id, content, source FROM documents ORDER BY id DESC",
        -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Document doc;
            doc.id = sqlite3_column_int(stmt, 0);
            auto* c = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            auto* s = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            if (c) doc.content = c; if (s) doc.source = s;
            results.push_back(doc);
        }
        sqlite3_finalize(stmt);
    }
    return results;
}

bool RAGManager::switchDevice(DeviceType device) {
    if (!initialized || device == currentDevice) return device == currentDevice;
    std::lock_guard<std::mutex> lock(m_mutex);
#ifdef FAISS_GPU_ENABLED
    if (device == DeviceType::GPU && checkGPUAvailability() && createGPUIndex()) {
        if (cpuIndex->ntotal > 0) loadIndexFromDatabase();
        currentIndex = gpuIndex; currentDevice = DeviceType::GPU;
        return true;
    }
#endif
    if (device == DeviceType::CPU) {
        currentIndex = cpuIndex; currentDevice = DeviceType::CPU; return true;
    }
    return false;
}

std::string RAGManager::getIndexInfo() const {
    if (!initialized || !currentIndex) return "Not initialized";
    std::string info = "Index: ";
    switch (indexType) {
    case IndexType::FLAT: info += "Flat"; break;
    case IndexType::IVF: info += "IVF"; break;
    case IndexType::SQ: info += "SQ"; break;
    }
    info += " | Device: " + std::string(currentDevice == DeviceType::GPU ? "GPU" : "CPU");
    info += " | Vectors: " + std::to_string(currentIndex->ntotal);
    info += " | Dim: " + std::to_string(dimension);
    info += " | Docs: " + std::to_string(getDocumentCount());
    info += " | Embedding: " + std::string(embeddingEngine ? "OK" : "MISSING");
    info += " | Reranker: " + std::string(reranker ? "OK" : "MISSING");
    return info;
}

std::string RAGManager::getDeviceStatus() const {
    std::string s = "Device: " + std::string(currentDevice == DeviceType::GPU ? "GPU" : "CPU");
#ifdef FAISS_GPU_ENABLED
    s += " | GPU: available";
#else
    s += " | GPU: not compiled";
#endif
    if (currentIndex) s += " | Vectors: " + std::to_string(currentIndex->ntotal);
    return s;
}