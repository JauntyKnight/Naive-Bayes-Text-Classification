#ifndef NAIVE_BAYES_TEXT_CLASSIFICATION_TF_IDF_VECTORIZER_H
#define NAIVE_BAYES_TEXT_CLASSIFICATION_TF_IDF_VECTORIZER_H

#include <unordered_map>
#include <string>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <limits>

class tf_idf_vectorizer {
public:
    tf_idf_vectorizer();
    tf_idf_vectorizer(
            std::string input="content",
            bool lowercase=true,
            std::string analyzer="word",
            std::unordered_set<std::string> stop_words={},
            std::string token_pattern="(?u)\\b\\w\\w+\\b",
            pair<int, int> ngram_range={1, 1},
            int alphabet_size=-1,
            int max_df=1.0,
            int min_df=0.0,
            int max_features=std::numeric_limits<size_t>,
            std::unordered_set<std::string> vocabulary={},
            bool use_idf=true,
            bool smooth_idf=true,
            bool sublinear_tf=false,
            ) {
        this->input = input;
        this->lowercase = lowercase;
        this->analyzer = analyzer;
        this->stop_words = stop_words;
        this->token_pattern = token_pattern;
        this->ngram_range = ngram_range;
        this->alphabet_size = alphabet_size;
        this->max_df = max_df;
        this->min_df = min_df;
        this->max_features = max_features;
        this->vocabulary = vocabulary;
        this->use_idf = use_idf;
        this->smooth_idf = smooth_idf;
        this->sublinear_tf = sublinear_tf;
    }

    template<typename Iter>
    void fit(const Iter &documents_begin, const Iter &documents_end) {
        // build alphabet
        build_alphabet(documents_begin, documents_end);

        for (auto it = documents_begin; it != documents_end; ++it) {
            std::string content = get_document_content(*it);

            if (lowercase) {
                std::transform(content.begin(), content.end(), content.begin(), std::tolower);
            }


            auto words_begin = std::sregex_iterator(content.begin(), content.end(), re);
            auto words_end = std::sregex_iterator();


        }
    }

    template<typename Iter>
    void fit_transform(const Iter &begin, const Iter &end);
    template<typename Iter>
    void transform(const Iter &begin, const Iter &end);
private:
    // params
    std::string input;
    std::unordered_map<std::string, int> word_count;
    bool lowercase;
    std::string analyzer;
    std::unordered_set<std::string> stop_words;
    std::string token_pattern;
    pair<int, int> ngram_range;
    int alphabet_size;
    double max_df;
    double min_df;
    int max_features;
    std::unordered_set<std::string> vocabulary;
    bool use_idf;
    bool smooth_idf;
    bool sublinear_tf;

    // internal
    std::unordered_set<char> alphabet;

    template <typename Iter>
    void build_alphabet(const Iter &begin, const Iter &end) {
        // alphabet_size of -1 means all characters
        if (alphabet_size == -1) {
            return;
        }

        // iterate over all documents and count the frequency of all characters
        std::unordered_map<char, int> char_count;
        for (auto it = begin; it != end; ++it) {
            std::string content = get_document_content(*it);
            for (char c : content) {
                char_count[c]++;
            }
        }

        // sort the characters by frequency
        std::vector<std::pair<char, int>> sorted_char_count(char_count.begin(), char_count.end());
        std::sort(sorted_char_count.begin(), sorted_char_count.end(), [](auto a, auto b) {
            return a.second > b.second;
        });

        // add the most frequent characters to the alphabet
        for (int i = 0; i < std::min(alphabet_size, sorted_char_count.size()); ++i) {
            alphabet.insert(sorted_char_count[i].first);
        }
    }

    std::string get_document_content(const std::string &document) {
        if (input == "content") {
            return document;
        }

        if (input == "filename") {
            std::ifstream file(document);
            return std::string((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            file.close();
        }

        if (input == "file") {
            return std::string((std::istreambuf_iterator<char>(document)),
                               std::istreambuf_iterator<char>());
        }
    }

    std::string filter_by_alphabet(std::string content) {
        if (alphabet.empty()) {
            return content;
        }

        std::string filtered_content;
        for (char c : content) {
            if (alphabet.find(c) != alphabet.end()) {
                filtered_content += c;
            }
        }
        return filtered_content;
    }

    std::vector<std::string> tokenize(const std::string &content) {
        std::vector<std::string> tokens;


        return tokens
    }


};


#endif //NAIVE_BAYES_TEXT_CLASSIFICATION_TF_IDF_VECTORIZER_H
