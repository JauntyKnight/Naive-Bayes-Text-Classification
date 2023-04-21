#include "tf_idf_vectorizer.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>

using namespace std;

// write unit tests for the tf_idf_vectorizer class

void test_fit() {
    // test the fit method

    // create a vector of 10 strings
    std::vector<std::string> documents = {
            "this is a test",
            "this is another test",
            "this is a third test",
            "this is a fourth test",
            "this is a fifth test",
            "this is a sixth test",
            "this is a seventh test",
            "this is an eighth test",
            "this is a ninth test",
            "this is a tenth test"
    };

    // create a tf_idf_vectorizer object
    tf_idf_vectorizer vectorizer;

    vectorizer.fit(documents.begin(), documents.end());

    // test the fit method
    cout << vectorizer.getVocabulary_size() << endl;

    cout << "df vocabulary: " << endl;
    for (auto it = vectorizer.getLearned_vocabulary().begin(); it != vectorizer.getLearned_vocabulary().end(); ++it) {
        cout << it->first << " " << it->second << endl;
    }

    cout << "word indices: " << endl;
    for (auto it = vectorizer.getWord_indices().begin(); it != vectorizer.getWord_indices().end(); ++it) {
        cout << it->first << " " << it->second << endl;
    }
}

void test_transofrm() {
    // test the transform method

    // create a vector of 10 strings
    std::vector<std::string> documents = {
            "this is a test",
            "this is another test",
            "this is a third test",
            "this is a fourth test",
            "this is a fifth test",
            "this is a sixth test",
            "this is a seventh test",
            "this is an eighth test",
            "this is a ninth test",
            "this is a tenth test"
    };

    // create a tf_idf_vectorizer object
    tf_idf_vectorizer vectorizer;

    vectorizer.fit(documents.begin(), documents.end());

    // test the transform method
    std::vector<std::vector<double>> tf_idf_vectors = vectorizer.transform(documents.begin(), documents.end());

    // set the precision of the output
    cout << fixed;
    cout << setprecision(2);

    // print the contents of the tf_idf_vectors
    for (auto it = tf_idf_vectors.begin(); it != tf_idf_vectors.end(); ++it) {
        for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
            cout << *it2 << " ";
        }
        cout << endl;
    }
}

int main() {
    test_fit();
    test_transofrm();
}