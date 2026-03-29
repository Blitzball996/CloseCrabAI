#include "EmbeddingEngine.h"
#include <iostream>
#include <spdlog/spdlog.h>

EmbeddingEngine::EmbeddingEngine(const std::string& modelPath,
    const std::string& vocabPath,
    bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "Embedding")
{
    // ---- 加载 WordPiece tokenizer ----
    tokenizer = std::make_unique<WordPieceTokenizer>(vocabPath);
    spdlog::info("[Embedding] WordPiece tokenizer loaded ({} tokens)", tokenizer->vocabSize());

    sessionOptions.SetIntraOpNumThreads(4);

#ifdef USE_ONNX_GPU
    if (useGPU) {
        try {
            OrtCUDAProviderOptions cuda_options;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[Embedding] Using CUDA\n";
        }
        catch (const std::exception& e) {
            std::cout << "[Embedding] CUDA not available: " << e.what() << "\n";
        }
    }
#endif

    try {
        std::wstring modelPathWstr(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, modelPathWstr.c_str(), sessionOptions);
        std::cout << "[Embedding] Model loaded successfully\n";

        // 获取输出维度
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::TypeInfo typeInfo = session->GetOutputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();
            if (shape.size() >= 3) {
                dimension = static_cast<int>(shape[2]);
                std::cout << "[Embedding] Dimension: " << dimension << "\n";
            }
        }
        catch (...) {
            std::cout << "[Embedding] Using default dimension: " << dimension << "\n";
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> EmbeddingEngine::encode(const std::string& text) {
    // 1. 使用真正的 WordPiece tokenizer
    auto tokens = tokenizer->tokenizeSingle(text);
    std::vector<int64_t> shape = { 1, static_cast<int64_t>(tokens.size()) };

    // 2. 创建 attention_mask (全1)
    std::vector<int64_t> attention_mask(tokens.size(), 1);

    // 3. 创建 token_type_ids (全0，单句)
    std::vector<int64_t> token_type_ids(tokens.size(), 0);

    // 4. 创建输入张量
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

    // 5. 准备输入
    const char* input_names[] = { "input_ids", "attention_mask", "token_type_ids" };
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));
    input_tensors.push_back(std::move(token_type_ids_tensor));

    const char* output_names[] = { "last_hidden_state" };

    // 6. 运行推理
    auto output_tensors = session->Run(
        Ort::RunOptions{ nullptr },
        input_names, input_tensors.data(), input_tensors.size(),
        output_names, 1);

    // 7. 获取输出
    float* data = output_tensors[0].GetTensorMutableData<float>();
    auto typeInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape_out = typeInfo.GetShape();

    int64_t seq_len = shape_out[1];
    int64_t hidden_dim = shape_out[2];

    // 8. Mean pooling
    std::vector<float> embedding(hidden_dim, 0.0f);
    for (int64_t i = 0; i < seq_len; i++) {
        for (int64_t j = 0; j < hidden_dim; j++) {
            embedding[j] += data[i * hidden_dim + j];
        }
    }
    for (int64_t j = 0; j < hidden_dim; j++) {
        embedding[j] /= static_cast<float>(seq_len);
    }

    dimension = static_cast<int>(hidden_dim);
    return embedding;
}