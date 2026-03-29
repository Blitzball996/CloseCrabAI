#include "LLMEngine.h"
#include <spdlog/spdlog.h>
#include <llama.h>
#include <ggml.h>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>
#include "../ssd/SSDExpertStreamer.h"
#include <fstream>
#include <thread>


extern bool g_llama_log_enabled;

LLMEngine::LLMEngine(const std::string& modelPath, int cpuMoeLayers)
    : m_cpuMoeLayers(cpuMoeLayers) {
    // 初始化 llama 后端
    llama_backend_init();

    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (g_llama_log_enabled) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                fprintf(stderr, "%s", text);
            }
        }
        }, nullptr);

    // 加载模型参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;  // 自动检测 GPU
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    // 处理 n_cpu_moe
    if (m_cpuMoeLayers > 0) {
        static std::vector<llama_model_tensor_buft_override> overrides;
        static std::vector<std::string> patterns;

        overrides.clear();
        patterns.clear();
        patterns.reserve(m_cpuMoeLayers);
        overrides.reserve(m_cpuMoeLayers + 1);

        for (int i = 0; i < m_cpuMoeLayers; ++i) {
            // 匹配第 i 个专家块的所有相关张量（根据实际模型调整）
            std::string pattern = "blk\\." + std::to_string(i) + "\\.ffn_(gate|up|down)";
            patterns.push_back(pattern);
            overrides.push_back({ patterns.back().c_str(), ggml_backend_cpu_buffer_type() });
        }
        overrides.push_back({ nullptr, nullptr });
        model_params.tensor_buft_overrides = overrides.data();
    }

    spdlog::info("Loading model from: {}", modelPath);
    spdlog::info("GPU layers: auto (-1)");

    spdlog::info("CPU MoE layers: {}", m_cpuMoeLayers);

    // 加载模型
    model = llama_load_model_from_file(modelPath.c_str(), model_params);
    if (!model) {
        spdlog::error("Failed to load model from: {}", modelPath);
        return;
    }
    spdlog::info("Model loaded successfully");

    // 获取 vocab
    vocab = llama_model_get_vocab(model);
    if (!vocab) {
        spdlog::error("Failed to get vocab from model");
        llama_free_model(model);
        model = nullptr;
        return;
    }

    // 创建上下文参数
    llama_context_params ctx_params = llama_context_default_params();
    size_t modelSizeMB = llama_model_size(model) / 1024 / 1024;
    if (modelSizeMB > 50000) {
        // 超大模型（>50GB）：保守上下文
        ctx_params.n_ctx = 32768;
    }
    else if (modelSizeMB > 10000) {
        // 大模型（10-50GB）
        ctx_params.n_ctx = 65536;
    }
    else {
        // 小模型：可以给大上下文
        ctx_params.n_ctx = 131072;
    }
    ctx_params.n_batch = 8192;          // 增大 batch 加速 prefill
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_seq_max = 1;

    // ====== KV Cache 量化：内存减半，质量几乎无损 ======
    ctx_params.type_k = GGML_TYPE_Q8_0;   // Key 从 FP16 压到 Q8
    ctx_params.type_v = GGML_TYPE_Q8_0;   // Value 从 FP16 压到 Q8
    spdlog::info("KV cache quantization: Q8_0 (memory ~50% of FP16)");

    // ====== Flash Attention：减少注意力计算内存峰值 ======
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    spdlog::info("Flash Attention: enabled");

    // 创建上下文
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        spdlog::error("Failed to create context for model");
        llama_free_model(model);
        model = nullptr;
        return;
    }

    spdlog::info("LLMEngine initialized successfully");
    spdlog::info("  - Context size: {}", llama_n_ctx(ctx));
    spdlog::info("  - Model size: {} MB", llama_model_size(model) / 1024 / 1024);
    spdlog::info("  - Vocabulary size: {}", llama_n_vocab(vocab));
    spdlog::info("  - CPU threads: {}", ctx_params.n_threads);

    // 检查 GPU 使用情况
    if (model_params.n_gpu_layers > 0) {
        spdlog::info("  - GPU acceleration: ENABLED ({} layers on GPU)", model_params.n_gpu_layers);
    }
    else {
        spdlog::info("  - GPU acceleration: DISABLED (CPU only)");
    }
}

LLMEngine::~LLMEngine() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_free_model(model);
        model = nullptr;
    }
    llama_backend_free();
    spdlog::info("LLMEngine destroyed");
}

std::vector<int> LLMEngine::stringToTokens(const std::string& text) const {
    const int maxTokens = static_cast<int>(text.length()) + 10;
    std::vector<int> tokens(maxTokens);

    // 用 vocab
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.length()),
        tokens.data(), maxTokens, true, true);
    if (n < 0) {
        spdlog::error("Failed to tokenize: {}", text.substr(0, 50));
        return {};
    }
    tokens.resize(n);
    return tokens;
}

std::string LLMEngine::tokensToString(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        std::string piece;
        piece.resize(128);
        // 用 vocab
        int n = llama_token_to_piece(vocab, token, piece.data(), static_cast<int>(piece.size()), 0, true);
        if (n > 0) {
            piece.resize(n);
            result += piece;
        }
    }
    return result;
}

void LLMEngine::generateTokensStreaming(const std::vector<int>& inputTokens,
    int maxTokens,
    float temperature,
    std::function<void(const std::string&)> onToken) {

    // ====== 修复：用 llama_kv_cache_clear 代替销毁重建 context ======
    // 原来的做法：llama_free(ctx) + llama_new_context(...)  —— 极慢，KV cache 浪费
    // 现在的做法：只清空 KV cache，context 保留复用 —— 快 1000 倍
    llama_memory_clear(llama_get_memory(ctx), true);

    const int n_vocab = llama_n_vocab(vocab);
    const int n_batch = 2048;  // 每批最多处理的 token 数

    // ====== 输入截断保护 ======
    std::vector<int> tokens = inputTokens;
    int maxCtx = (int)llama_n_ctx(ctx);
    if ((int)tokens.size() > maxCtx - maxTokens - 4) {
        spdlog::warn("Input too long ({} tokens), truncating", tokens.size());
        int keepFront = 200;
        int keepBack = maxCtx - keepFront - maxTokens - 4;
        if (keepBack > 0 && (int)tokens.size() > keepFront + keepBack) {
            std::vector<int> truncated;
            truncated.insert(truncated.end(), tokens.begin(), tokens.begin() + keepFront);
            truncated.insert(truncated.end(), tokens.end() - keepBack, tokens.end());
            tokens = std::move(truncated);
        }
        else {
            tokens.resize(maxCtx - maxTokens - 4);
        }
        spdlog::info("Truncated to {} tokens", tokens.size());
    }

    // ====== Prefill 阶段：分批处理全部输入 token ======
    // 把长 prompt 分成多个 batch 送入，避免超过 n_batch 限制
    for (int pos = 0; pos < (int)tokens.size(); pos += n_batch) {
        int batchSize = std::min(n_batch, (int)tokens.size() - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, batchSize);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            spdlog::error("Prefill failed at pos {}/{}", pos, (int)tokens.size());
            return;
        }
    }

    // ====== Decode 阶段：逐 token 生成 ======
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < maxTokens; ++i) {
        const float* logits = llama_get_logits(ctx);

        int nextToken = -1;

        if (temperature <= 0.0f) {
            // 贪心采样
            nextToken = (int)(std::max_element(logits, logits + n_vocab) - logits);
        }
        else {
            // 温度采样
            std::vector<float> probs(n_vocab);
            float maxLogit = *std::max_element(logits, logits + n_vocab);
            float sum = 0.0f;
            for (int j = 0; j < n_vocab; ++j) {
                probs[j] = expf((logits[j] - maxLogit) / temperature);
                sum += probs[j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < n_vocab; ++j) {
                    probs[j] /= sum;
                }
            }

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            nextToken = dist(gen);
        }

        if (nextToken == llama_token_eos(vocab)) {
            break;
        }

        // 输出 token
        std::string piece;
        piece.resize(128);
        int n = llama_token_to_piece(vocab, nextToken, piece.data(), (int)piece.size(), 0, true);
        if (n > 0) {
            piece.resize(n);
            onToken(piece);
        }

        // ====== 关键：只送 1 个新 token ======
        // KV cache 已经记住了之前所有 token，不需要重新送入
        // 原来的代码每轮都送全部 token，等于没用 KV cache
        llama_batch batch = llama_batch_get_one(&nextToken, 1);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            spdlog::error("Decode failed at token {}", i);
            break;
        }
    }
}

std::string LLMEngine::generate(const std::string& prompt,
    const std::string& system,
    int maxTokens,
    float temperature) {
    if (!isLoaded()) {
        spdlog::error("LLMEngine not loaded");
        return "";
    }

    std::string fullPrompt;
    if (!system.empty()) {
        fullPrompt = "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    fullPrompt += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
    fullPrompt += "<|im_start|>assistant\n";

    std::vector<int> inputTokens = stringToTokens(fullPrompt);
    if (inputTokens.empty()) {
        return "";
    }

    // ====== 新增：防止 token 数超过上下文窗口导致崩溃 ======
    int maxCtx = (int)llama_n_ctx(ctx);
    if ((int)inputTokens.size() > maxCtx - 4) {
        spdlog::warn("Input too long: {} tokens (max context: {}), truncating",
            inputTokens.size(), maxCtx);
        // 保留开头的系统 prompt（约 200 token）+ 截取末尾
        int keepFront = 200;
        int keepBack = maxCtx - keepFront - 4;  // 留 4 个 token 给生成
        if (keepBack > 0 && (int)inputTokens.size() > keepFront + keepBack) {
            std::vector<int> truncated;
            truncated.insert(truncated.end(),
                inputTokens.begin(),
                inputTokens.begin() + keepFront);
            truncated.insert(truncated.end(),
                inputTokens.end() - keepBack,
                inputTokens.end());
            inputTokens = std::move(truncated);
            spdlog::info("Truncated to {} tokens", inputTokens.size());
        }
        else {
            inputTokens.resize(maxCtx - 4);
        }
    }
    // ====== 截断保护结束 ======

    //generateTokensStreaming(inputTokens, maxTokens, temperature, onToken);

    std::vector<int> allTokens = inputTokens;
    std::random_device rd;
    std::mt19937 gen(rd());

    // 用 vocab
    const int n_vocab = llama_n_vocab(vocab);

    for (int i = 0; i < maxTokens; ++i) {
        llama_batch batch;
        if (i == 0) {
            // 第一次：送入全部输入 token（prefill）
            batch = llama_batch_get_one(allTokens.data(), (int)allTokens.size());
        }
        else {
            // 后续：只送最后一个新 token（decode）
            batch = llama_batch_get_one(&allTokens.back(), 1);
        }

        int n_eval = llama_decode(ctx, batch);
        if (n_eval != 0) {
            spdlog::error("Failed to evaluate model");
            break;
        }

        const float* logits = llama_get_logits(ctx);

        int nextToken = -1;

        if (temperature <= 0.0f) {
            nextToken = std::max_element(logits, logits + n_vocab) - logits;
        }
        else {
            std::vector<float> probs(n_vocab);
            float sum = 0.0f;
            for (int j = 0; j < n_vocab; ++j) {
                probs[j] = expf(logits[j] / temperature);
                sum += probs[j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < n_vocab; ++j) {
                    probs[j] /= sum;
                }
            }
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            nextToken = dist(gen);
        }

        if (nextToken == llama_token_eos(vocab)) {
            break;
        }

        allTokens.push_back(nextToken);
    }

    if (allTokens.size() > inputTokens.size()) {
        std::vector<int> responseTokens(allTokens.begin() + inputTokens.size(), allTokens.end());
        return tokensToString(responseTokens);
    }

    return "";
}

// ============================================
// 新增：直接使用完整 prompt 的生成方法（不添加任何格式）
// ============================================

void LLMEngine::generateRaw(const std::string& fullPrompt,
    int maxTokens,
    float temperature,
    std::function<void(const std::string&)> onToken,
    std::function<void()> onComplete) {
    if (!isLoaded()) {
        spdlog::error("LLMEngine not loaded");
        if (onComplete) onComplete();
        return;
    }

    std::vector<int> inputTokens = stringToTokens(fullPrompt);
    if (inputTokens.empty()) {
        spdlog::error("Failed to tokenize prompt");
        if (onComplete) onComplete();
        return;
    }

    // 直接调用，不再销毁重建 context
    // generateTokensStreaming 内部会 llama_kv_cache_clear
    generateTokensStreaming(inputTokens, maxTokens, temperature, onToken);

    if (onComplete) onComplete();
}

std::string LLMEngine::generateRaw(const std::string& fullPrompt,
    int maxTokens,
    float temperature) {
    if (!isLoaded()) {
        spdlog::error("LLMEngine not loaded");
        return "";
    }

    std::vector<int> inputTokens = stringToTokens(fullPrompt);
    if (inputTokens.empty()) {
        return "";
    }

    std::vector<int> allTokens = inputTokens;
    std::random_device rd;
    std::mt19937 gen(rd());

    const int n_vocab = llama_n_vocab(vocab);

    for (int i = 0; i < maxTokens; ++i) {
        llama_batch batch = llama_batch_get_one(allTokens.data(), allTokens.size());
        int n_eval = llama_decode(ctx, batch);
        if (n_eval != 0) {
            spdlog::error("Failed to evaluate model");
            break;
        }

        const float* logits = llama_get_logits(ctx);

        int nextToken = -1;

        if (temperature <= 0.0f) {
            nextToken = std::max_element(logits, logits + n_vocab) - logits;
        }
        else {
            std::vector<float> probs(n_vocab);
            float sum = 0.0f;
            for (int j = 0; j < n_vocab; ++j) {
                probs[j] = expf(logits[j] / temperature);
                sum += probs[j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < n_vocab; ++j) {
                    probs[j] /= sum;
                }
            }
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            nextToken = dist(gen);
        }

        if (nextToken == llama_token_eos(vocab)) {
            break;
        }

        allTokens.push_back(nextToken);
    }

    if (allTokens.size() > inputTokens.size()) {
        std::vector<int> responseTokens(allTokens.begin() + inputTokens.size(), allTokens.end());
        return tokensToString(responseTokens);
    }

    return "";
}

void LLMEngine::generateStreaming(const std::string& prompt,
    const std::string& system,
    int maxTokens,
    float temperature,
    std::function<void(const std::string&)> onToken,
    std::function<void()> onComplete) {
    if (!isLoaded()) {
        spdlog::error("LLMEngine not loaded");
        if (onComplete) onComplete();
        return;
    }

    std::string fullPrompt;
    if (!system.empty()) {
        fullPrompt = "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    fullPrompt += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
    fullPrompt += "<|im_start|>assistant\n";

    std::vector<int> inputTokens = stringToTokens(fullPrompt);
    if (inputTokens.empty()) {
        if (onComplete) onComplete();
        return;
    }

    generateTokensStreaming(inputTokens, maxTokens, temperature, onToken);

    if (onComplete) onComplete();
}

std::string LLMEngine::getModelInfo() const {
    if (!model) return "No model loaded";

    std::string info;
    info += "Model size: " + std::to_string(llama_model_size(model) / 1024 / 1024) + " MB\n";
    info += "Context size: " + std::to_string(llama_n_ctx(ctx)) + "\n";
    info += "Vocabulary size: " + std::to_string(llama_n_vocab(vocab));
    return info;
}

int LLMEngine::countTokens(const std::string& text) const {
    if (!isLoaded()) return 0;

    // 使用临时 batch 来获取 token 数量
    std::vector<int> tokens = stringToTokens(text);
    return static_cast<int>(tokens.size());
}

bool LLMEngine::initSSDStreaming(const std::string& expertDir,
    size_t cacheSizeMB,
    size_t gpuCacheSizeMB) {
    if (!isLoaded()) {
        spdlog::error("Model must be loaded before initializing SSD streaming");
        return false;
    }

    SSDStreamerConfig config;
    config.expertDir = expertDir;
    config.cacheSizeMB = cacheSizeMB;
    config.gpuCacheSizeMB = gpuCacheSizeMB;
    config.useMemoryMap = true;
    config.ioThreads = 4;
    config.enablePrefetch = true;
    config.prefetchDepth = 1;

    // 尝试从 manifest.json 读取模型参数
    std::string manifestPath = expertDir + "/manifest.json";
    std::ifstream mf(manifestPath);
    if (mf.is_open()) {
        spdlog::info("Found expert manifest: {}", manifestPath);
        // 这里可以解析 JSON 来自动设置参数
        // 简化版：使用默认的 Qwen3.5-397B 参数
        mf.close();
    }

    // Qwen3.5-397B-A17B 的参数
    // 如果你的模型不同，请修改这些值
    config.numLayers = 60;
    config.numExperts = 128;     // 每层的路由专家数
    config.activeExperts = 4;    // 每个 token 激活 K=4 个
    config.sharedExperts = 1;    // 共享专家
    config.hiddenDim = 4096;
    config.quantBits = 4;        // Q4 量化
    config.groupSize = 128;

    m_ssdStreamer = std::make_unique<SSDExpertStreamer>();
    if (!m_ssdStreamer->init(config)) {
        spdlog::error("Failed to initialize SSD Expert Streamer");
        m_ssdStreamer.reset();
        return false;
    }

    spdlog::info("SSD Expert Streaming initialized successfully!");
    spdlog::info("{}", m_ssdStreamer->getStatusString());
    return true;
}

std::string LLMEngine::getSSDStreamerStatus() const {
    if (!m_ssdStreamer) return "SSD Streaming: not initialized";
    return m_ssdStreamer->getStatusString();
}

bool LLMEngine::isSSDStreamingEnabled() const {
    return m_ssdStreamer != nullptr && m_ssdStreamer->isInitialized();
}