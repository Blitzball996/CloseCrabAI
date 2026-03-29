#include "RerankerEngine.h"
#include <iostream>
#include <spdlog/spdlog.h>

RerankerEngine::RerankerEngine(const std::string& modelPath,
    const std::string& vocabPath,
    bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Reranker")
{
    // ---- 加载 WordPiece tokenizer ----
    tokenizer = std::make_unique<WordPieceTokenizer>(vocabPath);
    spdlog::info("[Reranker] WordPiece tokenizer loaded ({} tokens)", tokenizer->vocabSize());

    sessionOptions.SetIntraOpNumThreads(4);

#ifdef USE_ONNX_GPU
    if (useGPU) {
        try {
            OrtCUDAProviderOptions cuda_options;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            spdlog::info("[Reranker] Using CUDA GPU");
        }
        catch (const std::exception& e) {
            spdlog::warn("[Reranker] CUDA not available: {}", e.what());
        }
    }
#endif

    try {
        std::wstring modelPathWstr(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, modelPathWstr.c_str(), sessionOptions);
        spdlog::info("[Reranker] Model loaded successfully");
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
        throw;
    }
}

float RerankerEngine::score(const std::string& query, const std::string& doc) {
    // 1. 使用真正的 WordPiece tokenizer 做 pair 编码
    std::vector<int64_t> token_type_ids;
    auto tokens = tokenizer->tokenizePair(query, doc, token_type_ids);
    std::vector<int64_t> shape = { 1, static_cast<int64_t>(tokens.size()) };

    // 2. attention_mask (全1)
    std::vector<int64_t> attention_mask(tokens.size(), 1);

    // 3. 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.data(), tokens.size(),
        shape.data(), shape.size());

    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(),
        shape.data(), shape.size());

    Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, token_type_ids.data(), token_type_ids.size(),
        shape.data(), shape.size());

    // 4. 准备输入
    const char* input_names[] = { "input_ids", "attention_mask", "token_type_ids" };
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));
    input_tensors.push_back(std::move(token_type_ids_tensor));

    const char* output_names[] = { "logits" };

    // 5. 运行推理
    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        input_names, input_tensors.data(), input_tensors.size(),
        output_names, 1);

    // 6. BGE Reranker 输出 [batch, 1]
    float* data = outputs[0].GetTensorMutableData<float>();
    return data[0];
}