#pragma once
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "WordPieceTokenizer.h"

class EmbeddingEngine {
public:
    /// @param modelPath  ONNX 模型路径 (如 models/bge-small-zh/model.onnx)
    /// @param vocabPath  vocab.txt 路径 (如 models/bge-small-zh/vocab.txt)
    /// @param useGPU     是否尝试使用 CUDA
    EmbeddingEngine(const std::string& modelPath,
        const std::string& vocabPath,
        bool useGPU = true);

    std::vector<float> encode(const std::string& text);

    int getDimension() const { return dimension; }

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;

    std::unique_ptr<WordPieceTokenizer> tokenizer;
    int dimension = 768;
};