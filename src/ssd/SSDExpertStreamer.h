#pragma once
// ============================================================================
// SSDExpertStreamer - Windows SSD Expert Streaming for MoE Models
// ============================================================================
// Inspired by danveloper/Flash-MoE's SSD streaming approach.
// Flash-MoE streams expert weights from NVMe SSD on-demand using parallel
// pread() on macOS. This module implements the same concept for Windows
// using overlapped I/O, memory-mapped files, and an LRU cache.
//
// Core idea: For MoE models like Qwen3.5-397B, only K=4 experts (out of
// 128-512) are active per token per layer. Instead of loading all experts
// into RAM/VRAM, we stream only the active ones from SSD on demand.
//
// Architecture:
//   [GGUF on NVMe SSD] --pread/mmap--> [Expert LRU Cache in RAM]
//                                            |
//                                     [Active experts]
//                                            |
//                                     [GPU VRAM via CUDA]
//
// Usage with your existing LLMEngine:
//   1. After loading model, call ExpertStreamer::init() with model metadata
//   2. Before each MoE layer forward pass, call prefetchExperts()
//   3. Call getExpertWeights() to get the weights for active experts
//   4. Weights are either served from cache or streamed from SSD
// ============================================================================

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <list>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Forward declarations
struct ExpertWeightData;
struct LayerExpertInfo;
struct StreamerStats;

// ============================================================================
// Configuration
// ============================================================================
struct SSDStreamerConfig {
    // --- Model layout ---
    int numLayers = 60;           // Total transformer layers
    int numExperts = 128;         // Experts per MoE layer (e.g., 128 for Qwen3-MoE)
    int activeExperts = 4;        // K: experts activated per token (top-K routing)
    int sharedExperts = 1;        // Shared experts (always loaded, not streamed)
    int hiddenDim = 4096;         // Hidden dimension
    int expertDim = 0;            // Expert FFN intermediate dim (auto-detected if 0)

    // --- Quantization ---
    int quantBits = 4;            // 4-bit or 2-bit expert quantization
    int groupSize = 128;          // Quantization group size

    // --- Cache settings ---
    size_t cacheSizeMB = 4096;    // Expert LRU cache size in MB (default 4GB)
    bool enableGPUCache = true;   // Also maintain a smaller GPU-side cache
    size_t gpuCacheSizeMB = 1024; // GPU cache size in MB (uses VRAM)

    // --- I/O settings ---
    int ioThreads = 4;            // Parallel I/O threads for SSD reads
    bool useDirectIO = false;     // Bypass OS page cache (F_NOCACHE equivalent)
    // Set true if model >> available RAM
    bool useMemoryMap = true;     // Use mmap for expert file access
    size_t readAheadMB = 0;       // Read-ahead buffer (0 = disabled)

    // --- Expert file layout ---
    std::string expertDir = "";   // Directory containing packed expert files
    // e.g., "models/packed_experts/"
    std::string expertFilePattern = "layer_{layer}_expert_{expert}.bin";

    // --- Performance tuning ---
    bool enablePrefetch = true;   // Enable speculative prefetch of likely experts
    bool warmCacheOnInit = false; // Pre-load frequently used experts on startup
    int prefetchDepth = 1;        // How many layers ahead to prefetch
};

// ============================================================================
// Expert weight data returned to the caller
// ============================================================================
struct ExpertWeightData {
    const void* gateWeight = nullptr;  // Gate projection weights
    const void* upWeight = nullptr;    // Up projection weights
    const void* downWeight = nullptr;  // Down projection weights

    const void* gateScales = nullptr;  // Quantization scales
    const void* upScales = nullptr;
    const void* downScales = nullptr;

    size_t gateSize = 0;               // Size in bytes
    size_t upSize = 0;
    size_t downSize = 0;

    int layerIdx = -1;
    int expertIdx = -1;
    bool isOnGPU = false;              // True if pointers are CUDA device pointers
    bool isValid = false;
};

// ============================================================================
// Statistics for monitoring
// ============================================================================
struct StreamerStats {
    std::atomic<uint64_t> cacheHits{ 0 };
    std::atomic<uint64_t> cacheMisses{ 0 };
    std::atomic<uint64_t> ssdReads{ 0 };
    std::atomic<uint64_t> bytesRead{ 0 };
    std::atomic<uint64_t> prefetchHits{ 0 };
    std::atomic<uint64_t> prefetchMisses{ 0 };

    // Timing (microseconds)
    std::atomic<uint64_t> totalReadTimeUs{ 0 };
    std::atomic<uint64_t> totalCacheTimeUs{ 0 };
    std::atomic<uint64_t> peakReadTimeUs{ 0 };

    double getCacheHitRate() const {
        uint64_t total = cacheHits.load() + cacheMisses.load();
        return total > 0 ? (double)cacheHits.load() / total : 0.0;
    }

    double getAvgReadTimeMs() const {
        uint64_t reads = ssdReads.load();
        return reads > 0 ? (double)totalReadTimeUs.load() / reads / 1000.0 : 0.0;
    }

    double getThroughputGBs() const {
        uint64_t timeUs = totalReadTimeUs.load();
        if (timeUs == 0) return 0.0;
        return (double)bytesRead.load() / (1024.0 * 1024.0 * 1024.0) /
            ((double)timeUs / 1000000.0);
    }

    std::string toString() const;
    void reset();
};

// ============================================================================
// LRU Cache Entry
// ============================================================================
struct CacheEntry {
    int layerIdx;
    int expertIdx;
    std::vector<uint8_t> data;     // Raw weight data in RAM
    void* gpuData = nullptr;       // CUDA device pointer (if cached on GPU)
    size_t totalSize = 0;
    bool pinned = false;           // Pinned entries are never evicted (shared experts)

    // Offsets within data buffer
    size_t gateOffset = 0;
    size_t gateSize = 0;
    size_t upOffset = 0;
    size_t upSize = 0;
    size_t downOffset = 0;
    size_t downSize = 0;

    // Quantization metadata offsets
    size_t gateScalesOffset = 0;
    size_t upScalesOffset = 0;
    size_t downScalesOffset = 0;
};

// ============================================================================
// Main Streamer Class
// ============================================================================
class SSDExpertStreamer {
public:
    SSDExpertStreamer();
    ~SSDExpertStreamer();

    // --- Lifecycle ---

    // Initialize with config. Call after model is loaded.
    bool init(const SSDStreamerConfig& config);

    // Initialize from a GGUF model file - auto-detects expert layout
    // This is the recommended init path for your LLMEngine integration
    bool initFromGGUF(const std::string& ggufPath, const SSDStreamerConfig& baseConfig);

    // Shutdown and release all resources
    void shutdown();

    bool isInitialized() const { return m_initialized; }

    // --- Core API ---

    // Get weights for specific experts in a layer.
    // Called during MoE forward pass after router selects top-K experts.
    //
    // layerIdx: transformer layer index
    // expertIndices: which experts were selected by the router (size = K)
    // returns: vector of ExpertWeightData, one per requested expert
    std::vector<ExpertWeightData> getExpertWeights(
        int layerIdx,
        const std::vector<int>& expertIndices);

    // Prefetch experts for upcoming layers (call from a background thread)
    // routerLogits: the router's output logits for the current token
    // currentLayer: which layer we're currently processing
    void prefetchExperts(int currentLayer,
        const std::vector<std::vector<float>>& routerLogits);

    // --- Integration helpers ---

    // Get the byte offset of an expert within the GGUF file
    // Useful for setting up mmap regions
    int64_t getExpertFileOffset(int layerIdx, int expertIdx) const;

    // Get total size of one expert's weights (gate + up + down)
    size_t getExpertSize() const;

    // Pre-load shared experts (they're always active, so always in cache)
    bool preloadSharedExperts();

    // --- Monitoring ---
    const StreamerStats& getStats() const { return m_stats; }
    void resetStats() { m_stats.reset(); }
    std::string getStatusString() const;

    // --- GPU helpers ---

    // Upload expert weights to GPU. Returns device pointer.
    // Caller should NOT free the pointer - it's managed by the GPU cache.
    void* uploadToGPU(int layerIdx, int expertIdx);

    // Hint that we're done with GPU expert (allows eviction)
    void releaseGPUExpert(int layerIdx, int expertIdx);

private:
    // --- Internal types ---
    using CacheKey = uint64_t;  // (layerIdx << 32) | expertIdx

    static CacheKey makeCacheKey(int layer, int expert) {
        return ((uint64_t)layer << 32) | (uint64_t)(uint32_t)expert;
    }

    // LRU list: front = most recently used, back = least recently used
    using LRUList = std::list<CacheKey>;
    using LRUIterator = LRUList::iterator;

    // --- Data members ---
    SSDStreamerConfig m_config;
    bool m_initialized = false;
    StreamerStats m_stats;

    // Expert file layout metadata (from GGUF or directory scan)
    struct ExpertLocation {
        std::string filePath;      // File containing this expert
        int64_t offset = 0;        // Byte offset within file
        size_t size = 0;           // Total size of expert weights
        size_t gateSize = 0;
        size_t upSize = 0;
        size_t downSize = 0;
    };

    // [layerIdx][expertIdx] -> location info
    std::vector<std::vector<ExpertLocation>> m_expertLocations;

    // --- LRU Cache ---
    std::mutex m_cacheMutex;
    std::unordered_map<CacheKey, CacheEntry> m_cache;
    std::unordered_map<CacheKey, LRUIterator> m_cacheIterators;
    LRUList m_lruList;
    size_t m_currentCacheSize = 0;
    size_t m_maxCacheSize = 0;

    // --- GPU Cache ---
    std::mutex m_gpuCacheMutex;
    std::unordered_map<CacheKey, void*> m_gpuCache;
    LRUList m_gpuLruList;
    std::unordered_map<CacheKey, LRUIterator> m_gpuCacheIterators;
    size_t m_currentGPUCacheSize = 0;
    size_t m_maxGPUCacheSize = 0;

    // --- File handles ---
#ifdef _WIN32
    // Memory-mapped file handles
    struct MappedFile {
        HANDLE fileHandle = INVALID_HANDLE_VALUE;
        HANDLE mappingHandle = NULL;
        void* baseAddress = nullptr;
        size_t fileSize = 0;
    };
    std::unordered_map<std::string, MappedFile> m_mappedFiles;

    // Overlapped I/O thread pool
    HANDLE m_ioCompletionPort = NULL;
    std::vector<std::thread> m_ioThreads;
    std::atomic<bool> m_ioRunning{ false };
#else
    // POSIX mmap
    struct MappedFile {
        int fd = -1;
        void* baseAddress = nullptr;
        size_t fileSize = 0;
    };
    std::unordered_map<std::string, MappedFile> m_mappedFiles;
#endif

    // --- Internal methods ---

    // Cache operations
    CacheEntry* lookupCache(CacheKey key);
    bool insertCache(CacheKey key, CacheEntry&& entry);
    void evictLRU(size_t bytesNeeded);
    void touchLRU(CacheKey key);

    // I/O operations
    bool readExpertFromDisk(int layerIdx, int expertIdx, std::vector<uint8_t>& outData);
    bool readExpertMapped(int layerIdx, int expertIdx, std::vector<uint8_t>& outData);
    bool readExpertDirect(int layerIdx, int expertIdx, std::vector<uint8_t>& outData);

    // File management
    bool openFile(const std::string& path);
    void closeAllFiles();
    bool mapFile(const std::string& path);

    // Expert layout detection
    bool detectExpertLayout();
    bool parseGGUFExpertOffsets(const std::string& ggufPath);
    bool scanExpertDirectory(const std::string& dir);

    // I/O thread pool (Windows IOCP)
    void startIOThreads();
    void stopIOThreads();
    void ioWorkerThread();

    // GPU operations
    bool initGPUCache();
    void cleanupGPUCache();
    void* allocGPU(size_t size);
    void freeGPU(void* ptr);
    bool copyToGPU(void* dst, const void* src, size_t size);
    void evictGPULRU(size_t bytesNeeded);
};