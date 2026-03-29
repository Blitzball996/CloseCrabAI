#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/// 通用 HuggingFace tokenizer.json 解析器
/// 支持 WordPiece (BERT / bge-small-zh) 和 BPE (XLM-R / bge-reranker-base)
class HFTokenizer {
public:
    /// @param tokenizerJsonPath  tokenizer.json 的完整路径
    explicit HFTokenizer(const std::string& tokenizerJsonPath);

    /// 单句编码: [CLS] tokens [SEP]  (BERT)  或  <s> tokens </s>  (XLM-R)
    std::vector<int64_t> encodeSingle(const std::string& text) const;

    /// 句对编码:
    ///   BERT:  [CLS] query [SEP] doc [SEP],  typeIds: 0..0 1..1
    ///   XLM-R: <s> query </s></s> doc </s>,  typeIds: all 0
    std::vector<int64_t> encodePair(const std::string& textA,
        const std::string& textB,
        std::vector<int64_t>& tokenTypeIds) const;

    int vocabSize() const { return static_cast<int>(vocab_.size()); }

private:
    // --- tokenizer.json 中解析出来的配置 ---
    enum class ModelType { WORDPIECE, BPE };
    ModelType modelType_ = ModelType::WORDPIECE;

    // 词表: token string → id
    std::unordered_map<std::string, int64_t> vocab_;

    // WordPiece 用的连续前缀
    std::string continuingSubwordPrefix_ = "##";

    // BPE merge 规则 (按优先级排列)
    std::vector<std::pair<std::string, std::string>> merges_;

    // 特殊 token
    int64_t clsId_ = -1;   // [CLS] 或 <s>
    int64_t sepId_ = -1;   // [SEP] 或 </s>
    int64_t unkId_ = 0;
    int64_t padId_ = 0;

    // --- 内部方法 ---
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<std::string> wordpieceTokenize(const std::string& word) const;
    std::vector<std::string> bpeTokenize(const std::string& word) const;
    std::vector<int64_t> tokenize(const std::string& text) const;

    int64_t tokenToId(const std::string& token) const;

    // UTF-8 工具
    static bool isChinese(uint32_t cp);
    static std::string addSpaceAroundChinese(const std::string& text);
    static std::string toLowerAscii(const std::string& s);
};