#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>

class SkillChain {
public:
    struct Step {
        std::string skillName;
        std::map<std::string, std::string> params;
        std::string result;
        bool completed = false;
    };

    void addStep(const std::string& skillName, const std::map<std::string, std::string>& params) {
        steps.push_back({ skillName, params, "", false });
    }

    void setResult(size_t index, const std::string& result) {
        if (index < steps.size()) {
            steps[index].result = result;
            steps[index].completed = true;
        }
    }

    std::string getResult(size_t index) const {
        if (index < steps.size()) {
            return steps[index].result;
        }
        return "";
    }

    bool isCompleted() const {
        for (const auto& step : steps) {
            if (!step.completed) return false;
        }
        return true;
    }

    Step getCurrentStep() const {
        for (const auto& step : steps) {
            if (!step.completed) return step;
        }
        return { "", {}, "", true };
    }

    std::string getProgress() const {
        std::stringstream ss;
        for (size_t i = 0; i < steps.size(); ++i) {
            ss << "步骤 " << (i + 1) << ": " << steps[i].skillName;
            if (steps[i].completed) {
                ss << " ✓";
                if (!steps[i].result.empty()) {
                    ss << " (结果: " << steps[i].result.substr(0, 50) << "...)";
                }
            }
            else {
                ss << " ⏳";
            }
            ss << "\n";
        }
        return ss.str();
    }

private:
    std::vector<Step> steps;
};