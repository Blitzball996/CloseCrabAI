#include "Config.h"
#include <spdlog/spdlog.h>

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

YAML::Node Config::getNode(const std::string& key) const {
    spdlog::debug("getNode: parsing key '{}'", key);
    if (!root.IsMap()) {
        spdlog::debug("Root is not a map");
        return YAML::Node();
    }

    const YAML::Node* current = &root;
    size_t start = 0;
    size_t dot;
    do {
        dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        if (!current->IsMap() || !(*current)[part].IsDefined()) {
            spdlog::debug("getNode: part '{}' not found", part);
            return YAML::Node();
        }
        current = &((*current)[part]);
        start = dot + 1;
    } while (dot != std::string::npos);

    spdlog::debug("getNode: success, node is scalar? {}", current->IsScalar());
    return *current;
}

bool Config::load(const std::string& filename) {
    try {
        root = YAML::LoadFile(filename);
        spdlog::info("Config loaded from: {}", filename);

        // 打印根键
        if (root.IsMap()) {
            spdlog::info("Root keys after load:");
            for (auto it = root.begin(); it != root.end(); ++it) {
                spdlog::info("  - {}", it->first.as<std::string>());
            }
        }

        // 原有打印
        if (root["llm"]) {
            spdlog::info("llm node exists");
            if (root["llm"]["model_path"]) {
                spdlog::info("model_path = {}", root["llm"]["model_path"].as<std::string>());
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        return false;
    }
}

std::string Config::getString(const std::string& key, const std::string& defaultValue) const {
    YAML::Node node = getNode(key);
    spdlog::debug("getString: key={}, node.IsNull={}, node.IsScalar={}", key, node.IsNull(), node.IsScalar());
    if (node && !node.IsNull() && node.IsScalar()) {
        return node.as<std::string>();
    }
    return defaultValue;
}

int Config::getInt(const std::string& key, int defaultValue) const {
    YAML::Node node = getNode(key);
    if (node && !node.IsNull() && node.IsScalar()) {
        return node.as<int>();
    }
    return defaultValue;
}

bool Config::getBool(const std::string& key, bool defaultValue) const {
    YAML::Node node = getNode(key);
    if (node) {
        return node.as<bool>();
    }
    return defaultValue;
}

double Config::getDouble(const std::string& key, double defaultValue) const {
    YAML::Node node = getNode(key);
    if (node && !node.IsNull() && node.IsScalar()) {
        return node.as<double>();
    }
    return defaultValue;
}