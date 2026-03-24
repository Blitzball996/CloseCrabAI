#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

class Config {
public:
    static Config& getInstance();

    bool load(const std::string& filename);

    std::string getString(const std::string& key, const std::string& defaultValue = "") const;
    int getInt(const std::string& key, int defaultValue = 0) const;
    bool getBool(const std::string& key, bool defaultValue = false) const;
    double getDouble(const std::string& key, double defaultValue = 0.0) const;

    int getGpuLayers() const {
        return getInt("gpu.layers", -1);
    }

    int getCpuMoe() const {
        return getInt("gpu.cpu_moe", 0);
    }

    int getBatchSize() const {
        return getInt("gpu.batch_size", 512);
    }

    int getThreads() const {
        return getInt("gpu.threads", 0);
    }

private:
    Config() = default;
    YAML::Node root;

    // 支持点号分隔的嵌套键
    YAML::Node getNode(const std::string& key) const;
};