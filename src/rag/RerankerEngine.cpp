#include <iostream>
#include "RerankerEngine.h"

RerankerEngine::RerankerEngine(const std::string& modelPath, bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Reranker")
{
#ifdef USE_ONNX_GPU
    if (useGPU) {
        OrtCUDAProviderOptions cuda_options;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
#endif

    try {
        std::wstring modelPathWstr(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, modelPathWstr.c_str(), sessionOptions);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
    }
}

std::vector<int64_t> RerankerEngine::tokenizePair(
    const std::string& q, const std::string& d)
{
    std::vector<int64_t> tokens(256, 0);
    // 简化处理
    return tokens;
}

float RerankerEngine::score(const std::string& query, const std::string& doc) {
    auto tokens = tokenizePair(query, doc);

    std::vector<int64_t> shape = { 1, (int64_t)tokens.size() };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.data(), tokens.size(),
        shape.data(), shape.size()
    );

    const char* input_names[] = { "input_ids" };
    const char* output_names[] = { "logits" };

    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* data = outputs[0].GetTensorMutableData<float>();

    return data[0]; // relevance score
}