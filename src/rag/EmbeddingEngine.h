#pragma once
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "HFTokenizer.h"

class EmbeddingEngine {
public:
    /// @param modelPath          ONNX 模型路径 (model.onnx，同目录自动读 model.onnx_data)
    /// @param tokenizerJsonPath  tokenizer.json 路径
    /// @param useGPU             是否尝试使用 CUDA
    EmbeddingEngine(const std::string& modelPath,
        const std::string& tokenizerJsonPath,
        bool useGPU = true);

    std::vector<float> encode(const std::string& text);

    int getDimension() const { return dimension; }

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;

    std::unique_ptr<HFTokenizer> tokenizer;
    int dimension = 768;
};