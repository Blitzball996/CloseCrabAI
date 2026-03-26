#include "EmbeddingEngine.h"
#include <iostream>

EmbeddingEngine::EmbeddingEngine(const std::string& modelPath, bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Embedding")
{
    sessionOptions.SetIntraOpNumThreads(4);

#ifdef USE_ONNX_GPU
    if (useGPU) {
        OrtCUDAProviderOptions cuda_options;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[Embedding] Using CUDA\n";
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

std::vector<int64_t> EmbeddingEngine::tokenize(const std::string& text) {
    // ⚠️ 简化版（你后面要换 tokenizer）
    std::vector<int64_t> tokens(128, 0);
    for (size_t i = 0; i < text.size() && i < 128; i++) {
        tokens[i] = (int64_t)text[i];
    }
    return tokens;
}

std::vector<float> EmbeddingEngine::encode(const std::string& text) {
    auto tokens = tokenize(text);

    std::vector<int64_t> shape = { 1, (int64_t)tokens.size() };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        tokens.data(),
        tokens.size(),
        shape.data(),
        shape.size()
    );

    const char* input_names[] = { "input_ids" };
    const char* output_names[] = { "last_hidden_state" };

    auto output_tensors = session->Run(
        Ort::RunOptions{ nullptr },
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* data = output_tensors[0].GetTensorMutableData<float>();

    std::vector<float> embedding(dimension);

    // mean pooling
    for (int i = 0; i < dimension; i++) {
        embedding[i] = data[i];
    }

    return embedding;
}