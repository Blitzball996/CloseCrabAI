#pragma once
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

class EmbeddingEngine {
public:
    EmbeddingEngine(const std::string& modelPath, bool useGPU = true);

    std::vector<float> encode(const std::string& text);

    int getDimension() const { return dimension; }

private:
    std::vector<int64_t> tokenize(const std::string& text);

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;

    int dimension = 768;
};