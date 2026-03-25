#include "LLMEngine.h"
#include <spdlog/spdlog.h>
#include <llama.h>
#include <ggml.h>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>
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
    ctx_params.n_ctx = 65536;      // 增大上下文到 4096
    ctx_params.n_batch = 1024;
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_seq_max = 1;

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
    // 清除之前的 KV 缓存，开始新的生成
    // 重新创建上下文，清除所有状态（包括 KV 缓存）
    if (ctx) {
        llama_free(ctx);
    }

    // 使用与原上下文相同的参数重新创建
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 65536;    // 与原值保持一致
    ctx_params.n_batch = 1024;
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_seq_max = 1;

    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        spdlog::error("Failed to recreate context");
        if (onToken) onToken(""); // 或传递错误信息
        return;
    }

    std::vector<int> allTokens = inputTokens;
    std::random_device rd;
    std::mt19937 gen(rd());

    // 用 vocab
    const int n_vocab = llama_n_vocab(vocab);

    for (int i = 0; i < maxTokens; ++i) {
        // 评估模型
        llama_batch batch = llama_batch_get_one(allTokens.data(), allTokens.size());
        int n_eval = llama_decode(ctx, batch);
        if (n_eval != 0) {
            spdlog::error("Failed to evaluate model");
            break;
        }

        // 获取下一个 token 的 logits
        const float* logits = llama_get_logits(ctx);

        // 采样下一个 token
        int nextToken = -1;

        if (temperature <= 0.0f) {
            // 贪婪采样
            nextToken = std::max_element(logits, logits + n_vocab) - logits;
        }
        else {
            // 温度采样
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

        // 检查是否结束，用 vocab
        if (nextToken == llama_token_eos(vocab)) {
            break;
        }

        allTokens.push_back(nextToken);

        // 将新 token 转换为字符串并回调，用 vocab
        std::string piece;
        piece.resize(128);
        int n = llama_token_to_piece(vocab, nextToken, piece.data(), static_cast<int>(piece.size()), 0, true);
        if (n > 0) {
            piece.resize(n);
            onToken(piece);
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

    std::vector<int> allTokens = inputTokens;
    std::random_device rd;
    std::mt19937 gen(rd());

    // 用 vocab
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

    // 新版 API - 清除内存缓存
    // 重新创建上下文，清除所有状态（包括 KV 缓存）
    if (ctx) {
        llama_free(ctx);
    }
    
    // 使用与原上下文相同的参数重新创建
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 65536;    // 与原值保持一致
    ctx_params.n_batch = 1024;
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_seq_max = 1;
    
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        spdlog::error("Failed to recreate context");
        if (onToken) onToken(""); // 或传递错误信息
        return;
    }

    std::vector<int> inputTokens = stringToTokens(fullPrompt);
    if (inputTokens.empty()) {
        spdlog::error("Failed to tokenize prompt");
        if (onComplete) onComplete();
        return;
    }

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