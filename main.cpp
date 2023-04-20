#include "tf_idf_vectorizer.hpp"

int main() {
    std::vector<std::string> documents = {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    };
    tf_idf_vectorizer vectorizer;
    vectorizer.fit(documents.begin(), documents.end());
    return 0;
}