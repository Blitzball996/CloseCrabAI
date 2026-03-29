#pragma once
#include <string>
#include <vector>
#include <unordered_map>

/// 真正的 WordPiece tokenizer，读取 BGE / BERT 的 vocab.txt
class WordPieceTokenizer {
public:
    /// @param vocabPath  vocab.txt 的完整路径
    /// @param maxInputChars 单词最大字符数，超过则标记为 [UNK]
    explicit WordPieceTokenizer(const std::string& vocabPath,
        int maxInputChars = 200);

    /// 对单个句子做完整分词，返回 token id 序列（不含 [CLS]/[SEP]）
    std::vector<int64_t> tokenize(const std::string& text) const;

    /// 对 query-document pair 分词，返回:
    ///   [CLS] query_tokens [SEP] doc_tokens [SEP]
    /// 同时填充 tokenTypeIds（query 部分=0，doc 部分=1）
    std::vector<int64_t> tokenizePair(const std::string& query,
        const std::string& doc,
        std::vector<int64_t>& tokenTypeIds) const;

    /// 对单句分词并加上 [CLS] / [SEP]
    std::vector<int64_t> tokenizeSingle(const std::string& text) const;

    int vocabSize() const { return static_cast<int>(vocab_.size()); }

private:
    // ---------- 内部方法 ----------
    /// Basic tokenization: 按空格 / 标点切分，并做 Unicode 规范化
    std::vector<std::string> basicTokenize(const std::string& text) const;

    /// WordPiece 子词切分
    std::vector<std::string> wordpieceTokenize(const std::string& word) const;

    /// 判断是否为中文字符（CJK Unified Ideographs 等）
    static bool isChinese(uint32_t cp);

    /// 在中文字符两侧加空格，使后续按空格切分能正确处理
    static std::string tokenizeChinese(const std::string& text);

    /// 简单的 UTF-8 → codepoint 遍历工具
    static std::vector<uint32_t> utf8ToCodepoints(const std::string& s);
    static std::string codepointsToUtf8(const std::vector<uint32_t>& cps);

    // ---------- 数据 ----------
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t unkId_ = 100;   // [UNK]
    int64_t clsId_ = 101;   // [CLS]
    int64_t sepId_ = 102;   // [SEP]
    int64_t padId_ = 0;     // [PAD]
    int maxInputChars_ = 200;
};