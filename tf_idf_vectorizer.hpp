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

    // learn the vocabulary and df from the training set
    template<typename Iter>
    void fit(const Iter &documents_begin, const Iter &documents_end) {
        alphabet.clear();
        learned_vocabulary.clear();
        // build alphabet
        build_alphabet(documents_begin, documents_end);

        for (auto it = documents_begin; it != documents_end; ++it) {
            std::string content = get_document_content(*it);

            if (lowercase) {
                std::transform(content.begin(), content.end(), content.begin(), std::tolower);
            }

            std::vector<std::string> n_grams = get_n_grams(content);

            learn_vocabulary_df(n_grams);
        }
    }

    template <typename Iter>
    matrix transform(const Iter &documents_begin, const Iter &documents_end) {
        matrix result;
        for (auto it = documents_begin; it != documents_end; ++it) {
            std::string content = get_document_content(*it);

            if (lowercase) {
                std::transform(content.begin(), content.end(), content.begin(), std::tolower);
            }

            std::vector<std::string> n_grams = get_n_grams(content);

            result.push_back(get_document_vector(n_grams));
        }
        return result;
    }


    template<typename Iter>
    void fit_transform(const Iter &begin, const Iter &end);
    template<typename Iter>
    void transform(const Iter &begin, const Iter &end);
private:
    // params
    std::string input;
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
    const char UNKNOWN_CHAR = '_';

    // internal
    std::unordered_set<char> alphabet;
    std::unordered_map<std::string, int> learned_vocabulary;
    size_t nr_documents;

    using matrix = std::vector<std::vector<double>>;

    struct word_count_pair {
        int tf;
        int df;
    };

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
            } else {
                filtered_content += UNKNOWN_CHAR;
            }
        }

        return filtered_content;
    }

    // tokenize the content of a document
    std::vector<std::string> tokenize(const std::string &content) {
        std::vector<std::string> tokens;
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), re);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::string match_str = i->.str();
            if (
                    !match_str.empty()
                    && stop_words.find(match_str) == stop_words.end()
                    && (vocabulary.empty() || vocabulary.find(match_str) != vocabulary.end())
                ) {
                tokens.push_back(filter_by_alphabet(match_str));
            }
        }

        return tokens
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

        if (analyzer == "char_wb") {
            return get_ngrams_char_wb(filter_by_alphabet(content));
        }
    }
};


#endif //NAIVE_BAYES_TEXT_CLASSIFICATION_TF_IDF_VECTORIZER_H
