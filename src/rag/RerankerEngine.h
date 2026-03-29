#pragma once
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "HFTokenizer.h"

class RerankerEngine {
public:
    /// @param modelPath          ONNX 模型路径
    /// @param tokenizerJsonPath  tokenizer.json 路径
    /// @param useGPU             是否尝试使用 CUDA
    RerankerEngine(const std::string& modelPath,
        const std::string& tokenizerJsonPath,
        bool useGPU = true);
    ~RerankerEngine() = default;

    float score(const std::string& query, const std::string& doc);

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;

    std::unique_ptr<HFTokenizer> tokenizer;
};