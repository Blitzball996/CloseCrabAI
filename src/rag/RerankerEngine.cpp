#include <iostream>
#include "RerankerEngine.h"
#include <spdlog/spdlog.h>

RerankerEngine::RerankerEngine(const std::string& modelPath, bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Reranker")
{
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

std::vector<int64_t> RerankerEngine::tokenizePair(
    const std::string& q, const std::string& d)
{
    std::vector<int64_t> tokens;

    // [CLS] token (101)
    tokens.push_back(101);

    // Query tokens
    for (char c : q) {
        tokens.push_back(static_cast<int64_t>(c));
    }

    // [SEP] token (102)
    tokens.push_back(102);

    // Document tokens
    for (char c : d) {
        tokens.push_back(static_cast<int64_t>(c));
    }

    // [SEP] token
    tokens.push_back(102);

    return tokens;
}

float RerankerEngine::score(const std::string& query, const std::string& doc) {
    // 1. Tokenize
    auto tokens = tokenizePair(query, doc);
    std::vector<int64_t> shape = { 1, static_cast<int64_t>(tokens.size()) };

    // 2. 创建 attention_mask (全1)
    std::vector<int64_t> attention_mask(tokens.size(), 1);

    // 3. 创建 token_type_ids
    // 规则: query部分为0，document部分为1
    std::vector<int64_t> token_type_ids(tokens.size(), 0);

    // 找到第一个 [SEP] 的位置
    size_t first_sep_pos = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == 102) {  // [SEP] token
            first_sep_pos = i;
            break;
        }
    }

    // 第一个 [SEP] 之后的部分（包括第二个 [SEP]）设为 1
    for (size_t i = first_sep_pos + 1; i < tokens.size(); i++) {
        token_type_ids[i] = 1;
    }

    // 4. 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // input_ids 张量
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        tokens.data(),
        tokens.size(),
        shape.data(),
        shape.size()
    );

    // attention_mask 张量
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        attention_mask.data(),
        attention_mask.size(),
        shape.data(),
        shape.size()
    );

    // token_type_ids 张量
    Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        token_type_ids.data(),
        token_type_ids.size(),
        shape.data(),
        shape.size()
    );

    // 5. 准备输入
    const char* input_names[] = { "input_ids", "attention_mask", "token_type_ids" };
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));
    input_tensors.push_back(std::move(token_type_ids_tensor));

    const char* output_names[] = { "logits" };

    // 6. 运行推理
    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        1
    );

    // 7. 获取得分
    float* data = outputs[0].GetTensorMutableData<float>();

    // BGE Reranker 输出通常是 [batch, 1]，返回第一个值
    return data[0];
}