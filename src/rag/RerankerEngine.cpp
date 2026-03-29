#include "RerankerEngine.h"
#include <iostream>
#include <spdlog/spdlog.h>

RerankerEngine::RerankerEngine(const std::string& modelPath,
    const std::string& tokenizerJsonPath,
    bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Reranker")
{
    // ---- 加载 tokenizer.json ----
    tokenizer = std::make_unique<HFTokenizer>(tokenizerJsonPath);
    spdlog::info("[Reranker] Tokenizer loaded ({} tokens)", tokenizer->vocabSize());

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
    // 1. 用 HFTokenizer 做 pair 编码（自动处理 BERT / XLM-R 格式差异）
    std::vector<int64_t> token_type_ids;
    auto tokens = tokenizer->encodePair(query, doc, token_type_ids);
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

    // 4. 检测模型需要哪些输入
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();

    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_tensors;

    input_names.push_back("input_ids");
    input_tensors.push_back(std::move(input_ids_tensor));

    input_names.push_back("attention_mask");
    input_tensors.push_back(std::move(attention_mask_tensor));

    // 检查模型是否需要 token_type_ids
    for (size_t i = 0; i < numInputs; i++) {
        auto name = session->GetInputNameAllocated(i, allocator);
        if (std::string(name.get()) == "token_type_ids") {
            input_names.push_back("token_type_ids");
            input_tensors.push_back(std::move(token_type_ids_tensor));
            break;
        }
    }

    // 5. 检测输出名称
    size_t numOutputs = session->GetOutputCount();
    std::vector<std::string> outputNameStrs;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < numOutputs; i++) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        outputNameStrs.push_back(name.get());
    }
    for (auto& s : outputNameStrs) output_names.push_back(s.c_str());

    // 6. 运行推理
    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        input_names.data(), input_tensors.data(), input_tensors.size(),
        output_names.data(), output_names.size());

    // 7. 获取得分
    float* data = outputs[0].GetTensorMutableData<float>();
    auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
    auto outShape = outInfo.GetShape();

    // 输出可能是 [batch, 1] 或 [batch, 2]（二分类 logits）
    if (outShape.size() >= 2 && outShape[1] == 2) {
        return data[1];  // 取 positive class logit
    }
    return data[0];
}