#pragma once
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

class RerankerEngine {
public:
    RerankerEngine(const std::string& modelPath, bool useGPU = true);
    ~RerankerEngine() = default;

    float score(const std::string& query, const std::string& doc);

private:
    std::vector<int64_t> tokenizePair(const std::string& q, const std::string& d);

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;
};