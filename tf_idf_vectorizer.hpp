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
#include <regex>
#include <vector>
#include <cmath>

class tf_idf_vectorizer {
public:
    using matrix = std::vector<std::vector<double> >;

    tf_idf_vectorizer(
            std::string input="content",
            bool lowercase=true,
            std::string analyzer="word",
            std::unordered_set<std::string> stop_words={},
            std::string token_pattern="\\b\\w\\w+\\b",
            std::pair<int, int> ngram_range=std::make_pair(1, 1),
            size_t alphabet_size=std::numeric_limits<size_t>::max(),
            bool use_tf=true,
            int max_df=std::numeric_limits<int>::max(),
            int min_df=0,
            std::unordered_set<std::string> vocabulary={},
            bool use_idf=true,
            bool smooth_idf=true,
            bool sublinear_tf=false
    ) {
        this->input = input;
        this->lowercase = lowercase;

        if (analyzer != "word" && analyzer != "char" && analyzer != "char_wb") {
            throw std::invalid_argument("analyzer must be either 'word', 'char' or 'char_wb'");
        }
        this->analyzer = analyzer;
        this->stop_words = stop_words;
        this->token_pattern = token_pattern;
        if (ngram_range.first < 1 || ngram_range.second < 1) {
            throw std::invalid_argument("ngram_range must be positive");
        }
        if (ngram_range.first > ngram_range.second) {
            throw std::invalid_argument("ngram_range must be (min_n, max_n) with min_n <= max_n");
        }
        this->ngram_range = ngram_range;
        this->alphabet_size = alphabet_size;
        this->use_tf = use_tf;
        this->max_df = max_df;
        this->min_df = min_df;
        this->vocabulary = vocabulary;
        this->use_idf = use_idf;
        this->smooth_idf = smooth_idf;
        this->sublinear_tf = sublinear_tf;
    }

    // learn the vocabulary and df from the training set
    template<typename Iter>
    void fit(const Iter &documents_begin, const Iter &documents_end) {
        alphabet.clear();
        learned_vocabulary.clear();
        // build alphabet
        build_alphabet(documents_begin, documents_end);

        nr_documents = smooth_idf ? 1 : 0;

        for (auto it = documents_begin; it != documents_end; ++it) {
            nr_documents++;

            std::string content = get_document_content(*it);

            if (lowercase) {
                std::transform(content.begin(), content.end(), content.begin(), [](char c) { return std::tolower(c); });
            }

            std::vector<std::string> n_grams = get_ngrams(content);

            learn_vocabulary_df(n_grams);
        }

        if (use_idf) {
            filter_df();
        }

        learn_word_indices();
    }

    template <typename Iter>
    matrix transform(const Iter &documents_begin, const Iter &documents_end) {
        matrix result;
        for (auto it = documents_begin; it != documents_end; ++it) {
            std::string content = get_document_content(*it);

            if (lowercase) {
                std::transform(content.begin(), content.end(), content.begin(), [](char c) { return std::tolower(c); });
            }

            std::vector<std::string> n_grams = get_ngrams(content);

            result.push_back(get_document_vector(n_grams));
        }
        return result;
    }

    template <typename Iter>
    matrix fit_transform(const Iter &documents_begin, const Iter &documents_end) {
        fit(documents_begin, documents_end);
        return transform(documents_begin, documents_end);
    }

    // getters and setters for the parameters
    const std::string &get_input() const {
        return input;
    }

    void set_input(const std::string &input) {
        tf_idf_vectorizer::input = input;
    }

    bool is_lowercase() const {
        return lowercase;
    }

    void set_lowercase(bool lowercase) {
        tf_idf_vectorizer::lowercase = lowercase;
    }

    const std::string &get_analyzer() const {
        return analyzer;
    }

    void set_analyzer(const std::string &analyzer) {
        tf_idf_vectorizer::analyzer = analyzer;
    }

    const std::unordered_set<std::string> &get_stop_words() const {
        return stop_words;
    }

    void set_stop_words(const std::unordered_set<std::string> &stop_words) {
        tf_idf_vectorizer::stop_words = stop_words;
    }

    const std::string &get_token_pattern() const {
        return token_pattern;
    }

    void set_token_pattern(const std::string &token_pattern) {
        tf_idf_vectorizer::token_pattern = token_pattern;
    }

    const std::pair<int, int> &get_ngram_range() const {
        return ngram_range;
    }

    void set_ngram_range(const std::pair<int, int> &ngram_range) {
        tf_idf_vectorizer::ngram_range = ngram_range;
    }

    int get_alphabet_size() const {
        return alphabet_size;
    }

    void set_alphabet_size(int alphabet_size) {
        tf_idf_vectorizer::alphabet_size = alphabet_size;
    }

    int getMax_df() const {
        return max_df;
    }

    void setMax_df(int max_df) {
        tf_idf_vectorizer::max_df = max_df;
    }

    int getMin_df() const {
        return min_df;
    }

    void setMin_df(int min_df) {
        tf_idf_vectorizer::min_df = min_df;
    }

    bool isUse_tf() const {
        return use_tf;
    }

    void setUse_tf(bool use_tf) {
        tf_idf_vectorizer::use_tf = use_tf;
    }

    const std::unordered_set<std::string> &getVocabulary() const {
        return vocabulary;
    }

    void setVocabulary(const std::unordered_set<std::string> &vocabulary) {
        tf_idf_vectorizer::vocabulary = vocabulary;
    }

    bool isUse_idf() const {
        return use_idf;
    }

    void setUse_idf(bool use_idf) {
        tf_idf_vectorizer::use_idf = use_idf;
    }

    bool isSmooth_idf() const {
        return smooth_idf;
    }

    void setSmooth_idf(bool smooth_idf) {
        tf_idf_vectorizer::smooth_idf = smooth_idf;
    }

    bool isSublinear_tf() const {
        return sublinear_tf;
    }

    void setSublinear_tf(bool sublinear_tf) {
        tf_idf_vectorizer::sublinear_tf = sublinear_tf;
    }

    const std::unordered_map<std::string, int> &getLearned_vocabulary() const {
        return learned_vocabulary;
    }

    const std::unordered_map<std::string, int> &getWord_indices() const {
        return word_indices;
    }

    size_t getNr_documents() const {
        return nr_documents;
    }

    const std::unordered_set<char> &getAlphabet() const {
        return alphabet;
    }

    const char getUnknown_char() const {
        return unknown_char;
    }

    const size_t getVocabulary_size() const {
        return word_indices.size();
    }

private:
    // params
    std::string input;
    bool lowercase;
    std::string analyzer;
    std::unordered_set<std::string> stop_words;
    std::string token_pattern;
    std::pair<int, int> ngram_range;
    size_t alphabet_size;
    int max_df;
    int min_df;
    bool use_tf;
    std::unordered_set<std::string> vocabulary;
    bool use_idf;
    bool smooth_idf;
    bool sublinear_tf;
    char unknown_char = '_';

    // internal
    std::unordered_set<char> alphabet;
    std::unordered_map<std::string, int> learned_vocabulary;
    std::unordered_map<std::string, int> word_indices;
    size_t nr_documents;

    template <typename Iter>
    void build_alphabet(const Iter &begin, const Iter &end) {
        if (alphabet_size == std::numeric_limits<int>::max()) {
            return;
        }

        // iterate over all documents and count the frequency of all characters
        std::unordered_map<char, int> char_count;
        for (auto it = begin; it != end; ++it) {
            std::string content = get_document_content(*it);
            for (char c : content) {
                if (lowercase) {
                    c = std::tolower(c);
                }
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

        // input == "file" from now on
        // read the contents of the file
        std::ifstream file(document);
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();

        return content;
    }

    std::string filter_by_alphabet(std::string content) {
        if (alphabet.empty()) {
            return content;
        }

        std::string filtered_content;
        for (char c : content) {
            if (alphabet.find(c) != alphabet.end()) {
                filtered_content += c;
            } else {
                filtered_content += unknown_char;
            }
        }

        return filtered_content;
    }

    // tokenize the content of a document
    std::vector<std::string> tokenize(const std::string &content) {
        std::vector<std::string> tokens;
        std::regex re(token_pattern);
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), re);
        auto words_end = std::sregex_iterator();

        for (auto i = words_begin; i != words_end; ++i) {
            std::string match_str = i->str();
            if (
                    !match_str.empty()
                    && stop_words.find(match_str) == stop_words.end()
                    && (vocabulary.empty() || vocabulary.find(match_str) != vocabulary.end())
                    ) {
                tokens.push_back(filter_by_alphabet(match_str));
            }
        }

        return tokens;
    }

    std::vector<std::string> get_ngrams_word(const std::vector<std::string> &tokens) {
        std::vector<std::string> ngrams;
        for (int i = ngram_range.first; i <= ngram_range.second; ++i) {
            for (int j = 0; j < tokens.size() - i + 1; ++j) {
                std::string ngram;
                for (int k = 0; k < i; ++k) {
                    ngram += tokens[j + k];
                    if (k != i - 1) {
                        ngram += " ";
                    }
                }
                ngrams.push_back(ngram);
            }
        }

        return ngrams;
    }

    std::vector<std::string> get_ngrams_char(const std::string &content) {
        std::vector<std::string> ngrams;
        // iterating over the content first and then over the ngrams in hope to increase cache locality
        for (int i = 0; i < content.size(); ++i) {
            for (int j = ngram_range.first; j <= ngram_range.second; ++j) {
                // ngrams to the right
                if (i + j <= content.size()) {
                    ngrams.push_back(content.substr(i, j));
                }

                // ngrams to the left
                if (i - j >= 0) {
                    ngrams.push_back(content.substr(i - j, j));
                }
            }
        }

        return ngrams;
    }

    std::string get_ngram_char_wb_helper(const std::string &content, int start, int end) {
        std::string ngram;
        bool met_wb = false;
        for (int i = start; i != end; ++i) {
            if (content[i] == ' ') {
                met_wb = true;
            }

            ngram += met_wb ? ' ' : content[i];
        }

        return ngram;
    }

    std::vector<std::string> get_ngrams_char_wb(const std::string &content) {
        std::vector<std::string> ngrams;
        // iterating over the content first and then over the ngrams in hope to increase cache locality
        for (int i = 0; i < content.size(); ++i) {
            for (int j = ngram_range.first; j <= ngram_range.second; ++j) {
                // ngrams to the right
                if (i + j <= content.size()) {
                    ngrams.push_back(get_ngram_char_wb_helper(content, i, i + j));
                }

                // ngrams to the left
                if (i - j >= 0) {
                    ngrams.push_back(get_ngram_char_wb_helper(content, i - j, i));
                }
            }
        }

        return ngrams;
    }

    void learn_vocabulary_df(const std::vector<std::string> &n_grams) {
        std::unordered_set<std::string> encountered_n_grams;

        for (const std::string &n_gram : n_grams) {
            if (encountered_n_grams.find(n_gram) == encountered_n_grams.end()) {
                learned_vocabulary[n_gram]++;
                encountered_n_grams.insert(n_gram);
            }
        }
    }

    std::vector<std::string> get_ngrams(const std::string &content) {
        if (analyzer == "word") {
            return get_ngrams_word(tokenize(content));
        }

        if (analyzer == "char") {
            return get_ngrams_char(filter_by_alphabet(content));
        }

        // analyzer == "char_wb"
        return get_ngrams_char_wb(filter_by_alphabet(content));
    }

    void filter_df() {
        for (auto it = learned_vocabulary.begin(); it != learned_vocabulary.end();) {
            double df = (double) it->second / (double) nr_documents;
            if (df < min_df || df > max_df) {
                it = learned_vocabulary.erase(it);
            } else {
                ++it;
            }
        }
    }

    void learn_word_indices() {
        int index = 0;
        for (const auto &word : learned_vocabulary) {
            word_indices[word.first] = index++;
        }
    }

    std::vector<double> get_document_vector(const std::vector<std::string> &n_grams) {
        std::vector<double> tf_idf_vector(learned_vocabulary.size(), 0);
        std::unordered_map<std::string, int> n_gram_count;

        if (use_tf) {
            for (const std::string &n_gram: n_grams) {
                n_gram_count[n_gram]++;
            }

            for (const auto &n_gram: n_gram_count) {
                double tf = (double)n_gram.second / (double)n_grams.size();
                tf_idf_vector[word_indices[n_gram.first]] = sublinear_tf ? 1 + std::log(tf) : tf;
            }
        } else {
            std::fill(tf_idf_vector.begin(), tf_idf_vector.end(), 1);
        }

        if (use_idf) {
            for (const std::string &n_gram: n_grams) {
                tf_idf_vector[word_indices[n_gram]] *= (1 + std::log((double) nr_documents / ((double) learned_vocabulary[n_gram] + (smooth_idf ? 1 : 0))));
            }
        }

        return tf_idf_vector;
    }
};


#endif //NAIVE_BAYES_TEXT_CLASSIFICATION_TF_IDF_VECTORIZER_H
