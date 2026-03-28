// ============================================================================
// SSDExpertStreamer.cpp - Windows SSD Expert Streaming for MoE Models
// ============================================================================
// Windows implementation using:
//   - Memory-mapped files (CreateFileMapping/MapViewOfFile)
//   - Overlapped I/O with IOCP for parallel SSD reads
//   - FILE_FLAG_NO_BUFFERING for direct SSD access (optional)
//   - CUDA for GPU cache management
//
// Equivalent to Flash-MoE's:
//   - pread() -> ReadFile() with OVERLAPPED
//   - GCD dispatch groups -> IOCP thread pool
//   - mmap -> MapViewOfFile
//   - F_NOCACHE -> FILE_FLAG_NO_BUFFERING
//   - Metal LRU cache -> CUDA device memory LRU cache
// ============================================================================

#include "SSDExpertStreamer.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <thread>
#include <numeric>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

// Optional CUDA support
#ifdef USE_CUDA_EXPERT_CACHE
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        spdlog::error("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)
#endif

// ============================================================================
// StreamerStats
// ============================================================================

std::string StreamerStats::toString() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "Expert Streamer Stats:\n"
        "  Cache hits:      %llu (%.1f%%)\n"
        "  Cache misses:    %llu\n"
        "  SSD reads:       %llu\n"
        "  Bytes read:      %.2f GB\n"
        "  Avg read time:   %.2f ms\n"
        "  Peak read time:  %.2f ms\n"
        "  SSD throughput:  %.2f GB/s\n"
        "  Prefetch hits:   %llu\n",
        (unsigned long long)cacheHits.load(),
        getCacheHitRate() * 100.0,
        (unsigned long long)cacheMisses.load(),
        (unsigned long long)ssdReads.load(),
        (double)bytesRead.load() / (1024.0 * 1024.0 * 1024.0),
        getAvgReadTimeMs(),
        (double)peakReadTimeUs.load() / 1000.0,
        getThroughputGBs(),
        (unsigned long long)prefetchHits.load());
    return std::string(buf);
}

void StreamerStats::reset() {
    cacheHits = 0;
    cacheMisses = 0;
    ssdReads = 0;
    bytesRead = 0;
    prefetchHits = 0;
    prefetchMisses = 0;
    totalReadTimeUs = 0;
    totalCacheTimeUs = 0;
    peakReadTimeUs = 0;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

SSDExpertStreamer::SSDExpertStreamer() = default;

SSDExpertStreamer::~SSDExpertStreamer() {
    shutdown();
}

// ============================================================================
// Initialization
// ============================================================================

bool SSDExpertStreamer::init(const SSDStreamerConfig& config) {
    if (m_initialized) {
        spdlog::warn("SSDExpertStreamer already initialized");
        return true;
    }

    m_config = config;
    m_maxCacheSize = config.cacheSizeMB * 1024ULL * 1024ULL;
    m_maxGPUCacheSize = config.gpuCacheSizeMB * 1024ULL * 1024ULL;

    spdlog::info("=== SSD Expert Streamer Initializing ===");
    spdlog::info("  Layers: {}", config.numLayers);
    spdlog::info("  Experts per layer: {} (active: {}, shared: {})",
        config.numExperts, config.activeExperts, config.sharedExperts);
    spdlog::info("  Quantization: {}-bit, group size {}", config.quantBits, config.groupSize);
    spdlog::info("  RAM cache: {} MB", config.cacheSizeMB);
    spdlog::info("  GPU cache: {} MB ({})", config.gpuCacheSizeMB,
        config.enableGPUCache ? "enabled" : "disabled");
    spdlog::info("  I/O threads: {}", config.ioThreads);
    spdlog::info("  Direct I/O: {}", config.useDirectIO ? "yes" : "no");
    spdlog::info("  Memory map: {}", config.useMemoryMap ? "yes" : "no");

    // Resize expert location table
    m_expertLocations.resize(config.numLayers);
    for (auto& layer : m_expertLocations) {
        layer.resize(config.numExperts);
    }

    // Detect expert layout from files
    if (!config.expertDir.empty()) {
        if (!scanExpertDirectory(config.expertDir)) {
            spdlog::error("Failed to scan expert directory: {}", config.expertDir);
            return false;
        }
    }

    // Start I/O thread pool
    if (!config.useMemoryMap) {
        startIOThreads();
    }

    // Initialize GPU cache
    if (config.enableGPUCache) {
        if (!initGPUCache()) {
            spdlog::warn("GPU cache initialization failed, continuing CPU-only");
            m_config.enableGPUCache = false;
        }
    }

    // Pre-load shared experts (they're always needed)
    if (config.sharedExperts > 0) {
        preloadSharedExperts();
    }

    m_initialized = true;
    spdlog::info("=== SSD Expert Streamer Ready ===");
    spdlog::info("  Expert size: {:.2f} MB each", (double)getExpertSize() / (1024.0 * 1024.0));
    spdlog::info("  Cache can hold: {} experts",
        m_maxCacheSize / std::max(getExpertSize(), (size_t)1));

    return true;
}

bool SSDExpertStreamer::initFromGGUF(const std::string& ggufPath,
    const SSDStreamerConfig& baseConfig) {
    m_config = baseConfig;

    spdlog::info("Parsing GGUF for expert layout: {}", ggufPath);

    if (!parseGGUFExpertOffsets(ggufPath)) {
        spdlog::error("Failed to parse GGUF expert offsets");
        return false;
    }

    return init(m_config);
}

void SSDExpertStreamer::shutdown() {
    if (!m_initialized) return;

    spdlog::info("Shutting down SSD Expert Streamer...");
    spdlog::info("{}", m_stats.toString());

    // Stop I/O threads
    stopIOThreads();

    // Cleanup GPU cache
    if (m_config.enableGPUCache) {
        cleanupGPUCache();
    }

    // Close all mapped files
    closeAllFiles();

    // Clear caches
    {
        std::lock_guard<std::mutex> lock(m_cacheMutex);
        m_cache.clear();
        m_cacheIterators.clear();
        m_lruList.clear();
        m_currentCacheSize = 0;
    }

    m_initialized = false;
    spdlog::info("SSD Expert Streamer shut down");
}

// ============================================================================
// Core API: getExpertWeights
// ============================================================================

std::vector<ExpertWeightData> SSDExpertStreamer::getExpertWeights(
    int layerIdx,
    const std::vector<int>& expertIndices)
{
    std::vector<ExpertWeightData> results;
    results.reserve(expertIndices.size());

    for (int expertIdx : expertIndices) {
        ExpertWeightData data;
        data.layerIdx = layerIdx;
        data.expertIdx = expertIdx;

        CacheKey key = makeCacheKey(layerIdx, expertIdx);

        // 1. Try cache lookup
        auto start = std::chrono::high_resolution_clock::now();

        CacheEntry* cached = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            cached = lookupCache(key);
        }

        if (cached) {
            // Cache hit!
            m_stats.cacheHits++;

            data.gateWeight = cached->data.data() + cached->gateOffset;
            data.upWeight = cached->data.data() + cached->upOffset;
            data.downWeight = cached->data.data() + cached->downOffset;
            data.gateScales = cached->data.data() + cached->gateScalesOffset;
            data.upScales = cached->data.data() + cached->upScalesOffset;
            data.downScales = cached->data.data() + cached->downScalesOffset;
            data.gateSize = cached->gateSize;
            data.upSize = cached->upSize;
            data.downSize = cached->downSize;
            data.isOnGPU = (cached->gpuData != nullptr);
            data.isValid = true;

            if (data.isOnGPU) {
                // Return GPU pointers instead
                size_t offset = 0;
                data.gateWeight = (uint8_t*)cached->gpuData + cached->gateOffset;
                data.upWeight = (uint8_t*)cached->gpuData + cached->upOffset;
                data.downWeight = (uint8_t*)cached->gpuData + cached->downOffset;
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            m_stats.totalCacheTimeUs += us;
        }
        else {
            // Cache miss - need to read from SSD
            m_stats.cacheMisses++;

            std::vector<uint8_t> rawData;
            bool readOk = false;

            if (m_config.useMemoryMap) {
                readOk = readExpertMapped(layerIdx, expertIdx, rawData);
            }
            else {
                readOk = readExpertDirect(layerIdx, expertIdx, rawData);
            }

            if (!readOk || rawData.empty()) {
                spdlog::error("Failed to read expert L{}E{} from SSD", layerIdx, expertIdx);
                data.isValid = false;
                results.push_back(data);
                continue;
            }

            // Build cache entry
            CacheEntry entry;
            entry.layerIdx = layerIdx;
            entry.expertIdx = expertIdx;
            entry.data = std::move(rawData);
            entry.totalSize = entry.data.size();

            // Parse the expert weight layout within the buffer
            // Layout: [gate_weights | up_weights | down_weights | scales]
            const auto& loc = m_expertLocations[layerIdx][expertIdx];
            entry.gateOffset = 0;
            entry.gateSize = loc.gateSize;
            entry.upOffset = loc.gateSize;
            entry.upSize = loc.upSize;
            entry.downOffset = loc.gateSize + loc.upSize;
            entry.downSize = loc.downSize;

            // Scales follow the weights
            size_t weightsEnd = loc.gateSize + loc.upSize + loc.downSize;
            size_t scaleSize = (entry.totalSize - weightsEnd) / 3;
            entry.gateScalesOffset = weightsEnd;
            entry.upScalesOffset = weightsEnd + scaleSize;
            entry.downScalesOffset = weightsEnd + 2 * scaleSize;

            // Fill return data
            data.gateWeight = entry.data.data() + entry.gateOffset;
            data.upWeight = entry.data.data() + entry.upOffset;
            data.downWeight = entry.data.data() + entry.downOffset;
            data.gateScales = entry.data.data() + entry.gateScalesOffset;
            data.upScales = entry.data.data() + entry.upScalesOffset;
            data.downScales = entry.data.data() + entry.downScalesOffset;
            data.gateSize = entry.gateSize;
            data.upSize = entry.upSize;
            data.downSize = entry.downSize;
            data.isValid = true;

            // Upload to GPU if enabled
            if (m_config.enableGPUCache) {
                void* gpuPtr = uploadToGPU(layerIdx, expertIdx);
                if (gpuPtr) {
                    entry.gpuData = gpuPtr;
                    data.isOnGPU = true;
                    data.gateWeight = (uint8_t*)gpuPtr + entry.gateOffset;
                    data.upWeight = (uint8_t*)gpuPtr + entry.upOffset;
                    data.downWeight = (uint8_t*)gpuPtr + entry.downOffset;
                }
            }

            // Insert into cache
            {
                std::lock_guard<std::mutex> lock(m_cacheMutex);
                insertCache(key, std::move(entry));
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            m_stats.totalReadTimeUs += us;
            if (us > m_stats.peakReadTimeUs.load()) {
                m_stats.peakReadTimeUs = us;
            }
        }

        results.push_back(data);
    }

    return results;
}

// ============================================================================
// Prefetch
// ============================================================================

void SSDExpertStreamer::prefetchExperts(
    int currentLayer,
    const std::vector<std::vector<float>>& routerLogits)
{
    if (!m_config.enablePrefetch) return;

    // Prefetch for the next N layers
    for (int ahead = 1; ahead <= m_config.prefetchDepth; ++ahead) {
        int targetLayer = currentLayer + ahead;
        if (targetLayer >= m_config.numLayers) break;

        // If we have router logits for this layer, use them to predict top-K
        if (targetLayer < (int)routerLogits.size() && !routerLogits[targetLayer].empty()) {
            const auto& logits = routerLogits[targetLayer];

            // Find top-K expert indices
            std::vector<int> indices(logits.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(),
                indices.begin() + m_config.activeExperts,
                indices.end(),
                [&logits](int a, int b) {
                    return logits[a] > logits[b];
                });

            std::vector<int> topK(indices.begin(),
                indices.begin() + m_config.activeExperts);

            // Prefetch these experts (background load into cache)
            for (int expertIdx : topK) {
                CacheKey key = makeCacheKey(targetLayer, expertIdx);

                std::lock_guard<std::mutex> lock(m_cacheMutex);
                if (lookupCache(key) == nullptr) {
                    // Not in cache - read from SSD
                    std::vector<uint8_t> rawData;
                    bool ok = false;

                    if (m_config.useMemoryMap) {
                        ok = readExpertMapped(targetLayer, expertIdx, rawData);
                    }
                    else {
                        ok = readExpertDirect(targetLayer, expertIdx, rawData);
                    }

                    if (ok && !rawData.empty()) {
                        CacheEntry entry;
                        entry.layerIdx = targetLayer;
                        entry.expertIdx = expertIdx;
                        entry.totalSize = rawData.size();
                        entry.data = std::move(rawData);

                        const auto& loc = m_expertLocations[targetLayer][expertIdx];
                        entry.gateOffset = 0;
                        entry.gateSize = loc.gateSize;
                        entry.upOffset = loc.gateSize;
                        entry.upSize = loc.upSize;
                        entry.downOffset = loc.gateSize + loc.upSize;
                        entry.downSize = loc.downSize;

                        insertCache(key, std::move(entry));
                        m_stats.prefetchHits++;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Cache Operations
// ============================================================================

CacheEntry* SSDExpertStreamer::lookupCache(CacheKey key) {
    // Caller must hold m_cacheMutex
    auto it = m_cache.find(key);
    if (it == m_cache.end()) return nullptr;

    // Move to front of LRU
    touchLRU(key);
    return &it->second;
}

bool SSDExpertStreamer::insertCache(CacheKey key, CacheEntry&& entry) {
    // Caller must hold m_cacheMutex
    size_t entrySize = entry.totalSize;

    // Evict if necessary
    if (m_currentCacheSize + entrySize > m_maxCacheSize) {
        evictLRU(entrySize);
    }

    // If still not enough room after eviction, skip caching
    if (m_currentCacheSize + entrySize > m_maxCacheSize) {
        spdlog::warn("Cache full, cannot insert L{}E{} ({} bytes)",
            entry.layerIdx, entry.expertIdx, entrySize);
        return false;
    }

    // Insert
    m_lruList.push_front(key);
    m_cacheIterators[key] = m_lruList.begin();
    m_cache[key] = std::move(entry);
    m_currentCacheSize += entrySize;

    return true;
}

void SSDExpertStreamer::evictLRU(size_t bytesNeeded) {
    // Caller must hold m_cacheMutex
    while (m_currentCacheSize + bytesNeeded > m_maxCacheSize && !m_lruList.empty()) {
        CacheKey victimKey = m_lruList.back();
        auto it = m_cache.find(victimKey);
        if (it != m_cache.end()) {
            // Don't evict pinned entries (shared experts)
            if (it->second.pinned) {
                // Move pinned entry to front so we skip it
                m_lruList.splice(m_lruList.begin(), m_lruList, std::prev(m_lruList.end()));
                continue;
            }

            // Free GPU memory if allocated
            if (it->second.gpuData) {
                freeGPU(it->second.gpuData);
                it->second.gpuData = nullptr;
            }

            m_currentCacheSize -= it->second.totalSize;
            m_cacheIterators.erase(victimKey);
            m_cache.erase(it);
        }
        m_lruList.pop_back();
    }
}

void SSDExpertStreamer::touchLRU(CacheKey key) {
    // Caller must hold m_cacheMutex
    auto it = m_cacheIterators.find(key);
    if (it != m_cacheIterators.end()) {
        m_lruList.splice(m_lruList.begin(), m_lruList, it->second);
    }
}

// ============================================================================
// SSD I/O: Memory-Mapped Read
// ============================================================================

bool SSDExpertStreamer::readExpertMapped(int layerIdx, int expertIdx,
    std::vector<uint8_t>& outData)
{
    if (layerIdx >= (int)m_expertLocations.size() ||
        expertIdx >= (int)m_expertLocations[layerIdx].size()) {
        return false;
    }

    const auto& loc = m_expertLocations[layerIdx][expertIdx];
    if (loc.filePath.empty() || loc.size == 0) return false;

    // Ensure file is mapped
    if (m_mappedFiles.find(loc.filePath) == m_mappedFiles.end()) {
        if (!mapFile(loc.filePath)) {
            spdlog::error("Failed to map file: {}", loc.filePath);
            return false;
        }
    }

    auto& mf = m_mappedFiles[loc.filePath];
    if (!mf.baseAddress) return false;

    // Bounds check
    if (loc.offset + loc.size > mf.fileSize) {
        spdlog::error("Expert L{}E{} extends beyond file (offset={}, size={}, fileSize={})",
            layerIdx, expertIdx, loc.offset, loc.size, mf.fileSize);
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Copy from mapped region
    // On Windows with NVMe, this triggers page faults that are served by
    // the SSD at ~7+ GB/s. The OS manages the page cache automatically.
    const uint8_t* src = (const uint8_t*)mf.baseAddress + loc.offset;
    outData.resize(loc.size);
    std::memcpy(outData.data(), src, loc.size);

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    m_stats.ssdReads++;
    m_stats.bytesRead += loc.size;
    m_stats.totalReadTimeUs += us;

    return true;
}

// ============================================================================
// SSD I/O: Direct (Non-Mapped) Read with Overlapped I/O
// ============================================================================

bool SSDExpertStreamer::readExpertDirect(int layerIdx, int expertIdx,
    std::vector<uint8_t>& outData)
{
    if (layerIdx >= (int)m_expertLocations.size() ||
        expertIdx >= (int)m_expertLocations[layerIdx].size()) {
        return false;
    }

    const auto& loc = m_expertLocations[layerIdx][expertIdx];
    if (loc.filePath.empty() || loc.size == 0) return false;

    auto start = std::chrono::high_resolution_clock::now();

#ifdef _WIN32
    // Windows: Use ReadFile with overlapped I/O for async SSD reads
    DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;
    if (m_config.useDirectIO) {
        flags |= FILE_FLAG_NO_BUFFERING;
    }

    HANDLE hFile = CreateFileA(
        loc.filePath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        flags,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        spdlog::error("Failed to open file: {} (error: {})",
            loc.filePath, GetLastError());
        return false;
    }

    // Align buffer for direct I/O
    size_t alignedSize = loc.size;
    if (m_config.useDirectIO) {
        // Direct I/O requires sector-aligned buffers and sizes
        alignedSize = (loc.size + 4095) & ~4095ULL;
    }

    // Allocate aligned buffer
    void* alignedBuf = nullptr;
    if (m_config.useDirectIO) {
        alignedBuf = _aligned_malloc(alignedSize, 4096);
    }
    else {
        outData.resize(loc.size);
        alignedBuf = outData.data();
    }

    if (!alignedBuf) {
        CloseHandle(hFile);
        return false;
    }

    // Set up overlapped structure for async read
    OVERLAPPED overlapped = {};
    overlapped.Offset = (DWORD)(loc.offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = (DWORD)(loc.offset >> 32);
    overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    DWORD bytesRead = 0;
    BOOL result = ReadFile(hFile, alignedBuf, (DWORD)alignedSize, &bytesRead, &overlapped);

    if (!result && GetLastError() == ERROR_IO_PENDING) {
        // Wait for async completion
        WaitForSingleObject(overlapped.hEvent, INFINITE);
        GetOverlappedResult(hFile, &overlapped, &bytesRead, FALSE);
    }

    CloseHandle(overlapped.hEvent);
    CloseHandle(hFile);

    if (m_config.useDirectIO) {
        outData.resize(loc.size);
        std::memcpy(outData.data(), alignedBuf, loc.size);
        _aligned_free(alignedBuf);
    }

#else
    // POSIX: Use pread() for direct reads (same as Flash-MoE)
    int fd = open(loc.filePath.c_str(), O_RDONLY);
    if (fd < 0) {
        spdlog::error("Failed to open file: {}", loc.filePath);
        return false;
    }

    outData.resize(loc.size);
    ssize_t bytesRead = pread(fd, outData.data(), loc.size, loc.offset);
    close(fd);

    if (bytesRead != (ssize_t)loc.size) {
        spdlog::error("Short read: expected {}, got {}", loc.size, bytesRead);
        return false;
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    m_stats.ssdReads++;
    m_stats.bytesRead += loc.size;
    m_stats.totalReadTimeUs += us;

    return true;
}

// ============================================================================
// File Management
// ============================================================================

bool SSDExpertStreamer::mapFile(const std::string& path) {
    if (m_mappedFiles.find(path) != m_mappedFiles.end()) {
        return true; // Already mapped
    }

#ifdef _WIN32
    MappedFile mf;

    mf.fileHandle = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL);

    if (mf.fileHandle == INVALID_HANDLE_VALUE) {
        spdlog::error("Failed to open file for mapping: {} (error: {})",
            path, GetLastError());
        return false;
    }

    // Get file size
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(mf.fileHandle, &fileSize)) {
        CloseHandle(mf.fileHandle);
        return false;
    }
    mf.fileSize = (size_t)fileSize.QuadPart;

    // Create file mapping
    mf.mappingHandle = CreateFileMappingA(
        mf.fileHandle,
        NULL,
        PAGE_READONLY,
        fileSize.HighPart,
        fileSize.LowPart,
        NULL);

    if (!mf.mappingHandle) {
        spdlog::error("Failed to create file mapping: {} (error: {})",
            path, GetLastError());
        CloseHandle(mf.fileHandle);
        return false;
    }

    // Map view
    mf.baseAddress = MapViewOfFile(
        mf.mappingHandle,
        FILE_MAP_READ,
        0, 0, 0);

    if (!mf.baseAddress) {
        spdlog::error("Failed to map view of file: {} (error: {})",
            path, GetLastError());
        CloseHandle(mf.mappingHandle);
        CloseHandle(mf.fileHandle);
        return false;
    }

    spdlog::info("Mapped file: {} ({:.2f} GB)",
        path, (double)mf.fileSize / (1024.0 * 1024.0 * 1024.0));

#else
    MappedFile mf;
    mf.fd = open(path.c_str(), O_RDONLY);
    if (mf.fd < 0) {
        spdlog::error("Failed to open file: {}", path);
        return false;
    }

    struct stat st;
    fstat(mf.fd, &st);
    mf.fileSize = st.st_size;

    mf.baseAddress = mmap(nullptr, mf.fileSize, PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (mf.baseAddress == MAP_FAILED) {
        close(mf.fd);
        return false;
    }

    // Advise random access pattern (like Flash-MoE)
    madvise(mf.baseAddress, mf.fileSize, MADV_RANDOM);

    spdlog::info("Mapped file: {} ({:.2f} GB)",
        path, (double)mf.fileSize / (1024.0 * 1024.0 * 1024.0));
#endif

    m_mappedFiles[path] = mf;
    return true;
}

void SSDExpertStreamer::closeAllFiles() {
    for (auto& [path, mf] : m_mappedFiles) {
#ifdef _WIN32
        if (mf.baseAddress) UnmapViewOfFile(mf.baseAddress);
        if (mf.mappingHandle) CloseHandle(mf.mappingHandle);
        if (mf.fileHandle != INVALID_HANDLE_VALUE) CloseHandle(mf.fileHandle);
#else
        if (mf.baseAddress && mf.baseAddress != MAP_FAILED) {
            munmap(mf.baseAddress, mf.fileSize);
        }
        if (mf.fd >= 0) close(mf.fd);
#endif
    }
    m_mappedFiles.clear();
}

// ============================================================================
// Expert Layout Detection
// ============================================================================

bool SSDExpertStreamer::scanExpertDirectory(const std::string& dir) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir)) {
        spdlog::error("Expert directory does not exist: {}", dir);
        return false;
    }

    int found = 0;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();

        // Parse filename: layer_N_expert_M.bin
        int layer = -1, expert = -1;
        if (sscanf(filename.c_str(), "layer_%d_expert_%d.bin", &layer, &expert) == 2) {
            if (layer >= 0 && layer < m_config.numLayers &&
                expert >= 0 && expert < m_config.numExperts) {

                auto& loc = m_expertLocations[layer][expert];
                loc.filePath = entry.path().string();
                loc.offset = 0;  // Entire file is one expert
                loc.size = (size_t)entry.file_size();

                // Estimate gate/up/down split based on quantization
                // For typical MoE: gate and up are [hidden_dim, expert_dim]
                //                   down is [expert_dim, hidden_dim]
                // At 4-bit: size = dim1 * dim2 / 2
                size_t weightSize = loc.size;
                // Approximate even split (actual split depends on model arch)
                loc.gateSize = weightSize / 3;
                loc.upSize = weightSize / 3;
                loc.downSize = weightSize - loc.gateSize - loc.upSize;

                found++;
            }
        }
    }

    spdlog::info("Found {} expert files in {}", found, dir);
    return found > 0;
}

bool SSDExpertStreamer::parseGGUFExpertOffsets(const std::string& ggufPath) {
    // Parse GGUF file to find expert tensor offsets
    // GGUF format: header -> metadata -> tensor info -> tensor data
    //
    // Expert tensors in Qwen3-MoE are named like:
    //   blk.{layer}.ffn_gate_exps.weight
    //   blk.{layer}.ffn_up_exps.weight
    //   blk.{layer}.ffn_down_exps.weight
    //
    // Each contains ALL experts for that layer packed together.
    // Individual expert offset = base_offset + expert_idx * single_expert_size

    std::ifstream file(ggufPath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Cannot open GGUF file: {}", ggufPath);
        return false;
    }

    // Read GGUF magic
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x46475547) {  // "GGUF" in little-endian
        spdlog::error("Not a valid GGUF file");
        return false;
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    spdlog::info("GGUF version: {}", version);

    // Read tensor count and metadata kv count
    uint64_t tensorCount, metadataKVCount;
    file.read(reinterpret_cast<char*>(&tensorCount), 8);
    file.read(reinterpret_cast<char*>(&metadataKVCount), 8);

    spdlog::info("GGUF: {} tensors, {} metadata entries", tensorCount, metadataKVCount);

    // For a full implementation, we would parse all metadata and tensor info
    // to find the exact byte offsets of each expert tensor.
    // This is model-specific and requires understanding the GGUF binary format.
    //
    // For now, we set up the location table assuming:
    // - Experts are stored in a single GGUF file
    // - The file is mmap'd, so we need byte offsets
    //
    // In practice, you'd use llama.cpp's own GGUF parser or the
    // `gguf-py` library to extract these offsets.

    // Store the GGUF path as the file path for all experts
    // The actual offsets would come from parsing the tensor info section
    for (int layer = 0; layer < m_config.numLayers; ++layer) {
        for (int expert = 0; expert < m_config.numExperts; ++expert) {
            auto& loc = m_expertLocations[layer][expert];
            loc.filePath = ggufPath;
            // These would be filled by proper GGUF parsing:
            loc.offset = 0;  // TODO: actual offset from GGUF tensor info
            loc.size = 0;    // TODO: actual size
        }
    }

    spdlog::warn("GGUF parsing is placeholder - use packed expert files for now");
    spdlog::info("Recommend: run extract_experts.py to create packed_experts/ directory");

    return true;
}

// ============================================================================
// Shared Expert Pre-loading
// ============================================================================

bool SSDExpertStreamer::preloadSharedExperts() {
    spdlog::info("Pre-loading {} shared expert(s) per layer...", m_config.sharedExperts);

    int loaded = 0;
    for (int layer = 0; layer < m_config.numLayers; ++layer) {
        // Shared expert is typically expert index 0 (or the last one)
        for (int se = 0; se < m_config.sharedExperts; ++se) {
            CacheKey key = makeCacheKey(layer, se);

            std::vector<uint8_t> rawData;
            bool ok = m_config.useMemoryMap
                ? readExpertMapped(layer, se, rawData)
                : readExpertDirect(layer, se, rawData);

            if (ok && !rawData.empty()) {
                CacheEntry entry;
                entry.layerIdx = layer;
                entry.expertIdx = se;
                entry.data = std::move(rawData);
                entry.totalSize = entry.data.size();
                entry.pinned = true;  // Never evict shared experts

                const auto& loc = m_expertLocations[layer][se];
                entry.gateOffset = 0;
                entry.gateSize = loc.gateSize;
                entry.upOffset = loc.gateSize;
                entry.upSize = loc.upSize;
                entry.downOffset = loc.gateSize + loc.upSize;
                entry.downSize = loc.downSize;

                std::lock_guard<std::mutex> lock(m_cacheMutex);
                insertCache(key, std::move(entry));
                loaded++;
            }
        }
    }

    spdlog::info("Pre-loaded {} shared experts", loaded);
    return loaded > 0;
}

// ============================================================================
// Helper Methods
// ============================================================================

size_t SSDExpertStreamer::getExpertSize() const {
    // Calculate size of one expert based on config
    int dim = m_config.hiddenDim;
    int expertDim = m_config.expertDim > 0 ? m_config.expertDim : dim * 4;

    // gate: [dim, expertDim], up: [dim, expertDim], down: [expertDim, dim]
    size_t elementsPerExpert = (size_t)dim * expertDim * 2 + (size_t)expertDim * dim;

    // Quantized size
    size_t bytesPerExpert = elementsPerExpert * m_config.quantBits / 8;

    // Add scale/bias overhead (~2% for group_size=128)
    size_t scaleOverhead = (elementsPerExpert / m_config.groupSize) * 4; // float32 scales
    bytesPerExpert += scaleOverhead;

    return bytesPerExpert;
}

int64_t SSDExpertStreamer::getExpertFileOffset(int layerIdx, int expertIdx) const {
    if (layerIdx >= (int)m_expertLocations.size() ||
        expertIdx >= (int)m_expertLocations[layerIdx].size()) {
        return -1;
    }
    return m_expertLocations[layerIdx][expertIdx].offset;
}

std::string SSDExpertStreamer::getStatusString() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "SSD Expert Streamer:\n"
        "  Initialized: %s\n"
        "  Cache: %.1f / %.1f MB (%.1f%%)\n"
        "  GPU Cache: %.1f / %.1f MB\n"
        "  Mapped files: %zu\n"
        "  Expert size: %.2f MB\n",
        m_initialized ? "yes" : "no",
        (double)m_currentCacheSize / (1024.0 * 1024.0),
        (double)m_maxCacheSize / (1024.0 * 1024.0),
        m_maxCacheSize > 0 ? (double)m_currentCacheSize / m_maxCacheSize * 100.0 : 0.0,
        (double)m_currentGPUCacheSize / (1024.0 * 1024.0),
        (double)m_maxGPUCacheSize / (1024.0 * 1024.0),
        m_mappedFiles.size(),
        (double)getExpertSize() / (1024.0 * 1024.0));
    return std::string(buf) + m_stats.toString();
}

// ============================================================================
// I/O Thread Pool (Windows IOCP)
// ============================================================================

void SSDExpertStreamer::startIOThreads() {
#ifdef _WIN32
    m_ioCompletionPort = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, m_config.ioThreads);
    if (!m_ioCompletionPort) {
        spdlog::error("Failed to create IOCP (error: {})", GetLastError());
        return;
    }

    m_ioRunning = true;
    for (int i = 0; i < m_config.ioThreads; ++i) {
        m_ioThreads.emplace_back(&SSDExpertStreamer::ioWorkerThread, this);
    }
    spdlog::info("Started {} I/O worker threads", m_config.ioThreads);
#endif
}

void SSDExpertStreamer::stopIOThreads() {
#ifdef _WIN32
    m_ioRunning = false;

    // Post completion packets to wake up threads
    if (m_ioCompletionPort) {
        for (size_t i = 0; i < m_ioThreads.size(); ++i) {
            PostQueuedCompletionStatus(m_ioCompletionPort, 0, 0, NULL);
        }
    }

    for (auto& t : m_ioThreads) {
        if (t.joinable()) t.join();
    }
    m_ioThreads.clear();

    if (m_ioCompletionPort) {
        CloseHandle(m_ioCompletionPort);
        m_ioCompletionPort = NULL;
    }
#endif
}

void SSDExpertStreamer::ioWorkerThread() {
#ifdef _WIN32
    while (m_ioRunning) {
        DWORD bytesTransferred = 0;
        ULONG_PTR completionKey = 0;
        LPOVERLAPPED pOverlapped = NULL;

        BOOL result = GetQueuedCompletionStatus(
            m_ioCompletionPort,
            &bytesTransferred,
            &completionKey,
            &pOverlapped,
            1000);  // 1 second timeout

        if (!result) {
            if (GetLastError() == WAIT_TIMEOUT) continue;
            if (!m_ioRunning) break;
        }

        if (pOverlapped == NULL) continue;  // Shutdown signal

        // Process completed I/O
        // The completion handler is embedded in the OVERLAPPED extension structure
        // (would be implemented for async prefetch operations)
    }
#endif
}

// ============================================================================
// GPU Cache Operations
// ============================================================================

bool SSDExpertStreamer::initGPUCache() {
#ifdef USE_CUDA_EXPERT_CACHE
    spdlog::info("Initializing GPU expert cache ({} MB)", m_config.gpuCacheSizeMB);

    // Verify CUDA is available
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        spdlog::warn("No CUDA devices found");
        return false;
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    spdlog::info("GPU: {} ({} MB free)", props.name,
        (int)(props.totalGlobalMem / 1024 / 1024));

    return true;
#else
    spdlog::info("GPU cache disabled (compile with -DUSE_CUDA_EXPERT_CACHE to enable)");
    return false;
#endif
}

void SSDExpertStreamer::cleanupGPUCache() {
#ifdef USE_CUDA_EXPERT_CACHE
    std::lock_guard<std::mutex> lock(m_gpuCacheMutex);
    for (auto& [key, ptr] : m_gpuCache) {
        if (ptr) cudaFree(ptr);
    }
    m_gpuCache.clear();
    m_gpuLruList.clear();
    m_gpuCacheIterators.clear();
    m_currentGPUCacheSize = 0;
#endif
}

void* SSDExpertStreamer::allocGPU(size_t size) {
#ifdef USE_CUDA_EXPERT_CACHE
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
#else
    return nullptr;
#endif
}

void SSDExpertStreamer::freeGPU(void* ptr) {
#ifdef USE_CUDA_EXPERT_CACHE
    if (ptr) cudaFree(ptr);
#endif
}

bool SSDExpertStreamer::copyToGPU(void* dst, const void* src, size_t size) {
#ifdef USE_CUDA_EXPERT_CACHE
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return err == cudaSuccess;
#else
    return false;
#endif
}

void* SSDExpertStreamer::uploadToGPU(int layerIdx, int expertIdx) {
#ifdef USE_CUDA_EXPERT_CACHE
    CacheKey key = makeCacheKey(layerIdx, expertIdx);

    std::lock_guard<std::mutex> gpuLock(m_gpuCacheMutex);

    // Check if already on GPU
    auto it = m_gpuCache.find(key);
    if (it != m_gpuCache.end()) {
        return it->second;
    }

    // Get from CPU cache
    CacheEntry* cpuEntry = nullptr;
    {
        std::lock_guard<std::mutex> cpuLock(m_cacheMutex);
        cpuEntry = lookupCache(key);
    }
    if (!cpuEntry) return nullptr;

    // Evict GPU entries if needed
    if (m_currentGPUCacheSize + cpuEntry->totalSize > m_maxGPUCacheSize) {
        evictGPULRU(cpuEntry->totalSize);
    }

    // Allocate and copy
    void* gpuPtr = allocGPU(cpuEntry->totalSize);
    if (!gpuPtr) return nullptr;

    if (!copyToGPU(gpuPtr, cpuEntry->data.data(), cpuEntry->totalSize)) {
        freeGPU(gpuPtr);
        return nullptr;
    }

    m_gpuCache[key] = gpuPtr;
    m_gpuLruList.push_front(key);
    m_gpuCacheIterators[key] = m_gpuLruList.begin();
    m_currentGPUCacheSize += cpuEntry->totalSize;

    return gpuPtr;
#else
    return nullptr;
#endif
}

void SSDExpertStreamer::releaseGPUExpert(int layerIdx, int expertIdx) {
    // Just a hint - we don't immediately free, just allow eviction
    // The LRU will handle actual eviction when space is needed
}

void SSDExpertStreamer::evictGPULRU(size_t bytesNeeded) {
#ifdef USE_CUDA_EXPERT_CACHE
    // Caller must hold m_gpuCacheMutex
    while (m_currentGPUCacheSize + bytesNeeded > m_maxGPUCacheSize &&
        !m_gpuLruList.empty()) {
        CacheKey victimKey = m_gpuLruList.back();

        auto it = m_gpuCache.find(victimKey);
        if (it != m_gpuCache.end()) {
            freeGPU(it->second);

            // Also clear the gpuData pointer in the CPU cache entry
            auto cpuIt = m_cache.find(victimKey);
            if (cpuIt != m_cache.end()) {
                cpuIt->second.gpuData = nullptr;
            }

            // Estimate size (we don't track per-entry GPU size separately)
            size_t entrySize = getExpertSize();
            m_currentGPUCacheSize = (m_currentGPUCacheSize > entrySize)
                ? m_currentGPUCacheSize - entrySize : 0;

            m_gpuCache.erase(it);
            m_gpuCacheIterators.erase(victimKey);
        }
        m_gpuLruList.pop_back();
    }
#endif
}