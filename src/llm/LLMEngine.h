#pragma once
#include <string>
#include <memory>
#include <vector>
#include <cstddef>
#include <functional>

// 前向声明
struct llama_model;
struct llama_context;
struct llama_vocab;
class SSDExpertStreamer;
struct SSDStreamerConfig;

class LLMEngine {
public:
    LLMEngine(const std::string& modelPath, int cpuMoeLayers = 0);
    ~LLMEngine();

    std::string generate(const std::string& prompt,
        const std::string& system = "",
        int maxTokens = 512,
        float temperature = 0.7f);

    void generateStreaming(const std::string& prompt,
        const std::string& system,
        int maxTokens,
        float temperature,
        std::function<void(const std::string&)> onToken,
        std::function<void()> onComplete = nullptr);

    // 新增：直接使用完整 prompt（不添加任何格式）
    std::string generateRaw(const std::string& fullPrompt,
        int maxTokens = 512,
        float temperature = 0.7f);

    void generateRaw(const std::string& fullPrompt,
        int maxTokens,
        float temperature,
        std::function<void(const std::string&)> onToken,
        std::function<void()> onComplete = nullptr);

    bool isLoaded() const { return model != nullptr && ctx != nullptr; }
    std::string getModelInfo() const;
    // 计算文本的 token 数量
    int countTokens(const std::string& text) const;

    // ====== SSD Expert Streaming ======
    // 初始化 SSD 流式专家加载（用于 MoE 大模型）
    bool initSSDStreaming(const std::string& expertDir,
        size_t cacheSizeMB = 4096,
        size_t gpuCacheSizeMB = 1024);

    // 获取 SSD Streamer 状态信息
    std::string getSSDStreamerStatus() const;

    // 是否启用了 SSD 流式加载
    bool isSSDStreamingEnabled() const;

private:
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    const struct llama_vocab* vocab = nullptr;  // 添加 vocab 成员
    int m_cpuMoeLayers = 0;

    std::vector<int> stringToTokens(const std::string& text) const;
    std::string tokensToString(const std::vector<int>& tokens) const;
    void generateTokensStreaming(const std::vector<int>& inputTokens,
        int maxTokens,
        float temperature,
        std::function<void(const std::string&)> onToken);

    // SSD Expert Streamer（MoE 大模型专用）
    std::unique_ptr<SSDExpertStreamer> m_ssdStreamer;
};