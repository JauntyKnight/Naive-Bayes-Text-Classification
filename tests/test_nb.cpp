// testing the gaussian naive bayes classifier from ../naive_bayes.hpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <utility>
#include <sstream>
#include <string>
#include <memory>
#include <cassert>
#include <algorithm>

#include "../naive_bayes.hpp"

using namespace std;

pair<vector<vector<double> >, vector<size_t> > read_dataset(const string &path) {
    vector<vector<double> > data;
    vector<size_t> labels;

    ifstream fin(path);
    if (!fin.is_open()) {
        cout << "Error opening file " << path << endl;
        exit(1);
    }

    string line;
    while (getline(fin, line)) {
        vector<double> row;
        size_t label;
        stringstream ss(line);
        double val;
        while (ss >> val) {
            row.push_back(val);
            ss.ignore();
        }
        label = row.back();
        row.pop_back();
        data.push_back(row);
        labels.push_back(round(label));
    }

    return make_pair(data, labels);
}


// void fit_predict(NaiveBayes &nb, vector<vector<double> > train_data, vector<size_t> train_labels, vector<vector<double> > test_data, vector<size_t> test_labels) {
//     nb.fit(train_data, train_labels);
    
//     auto predictions = nb.predict(test_data);
//     double correct = 0;

//     for (size_t i = 0; i < predictions.size(); ++i) {
//         if (predictions[i] == test_labels[i]) {
//             correct += 1;
//         }
//     }

//     cout << "Accuracy: " << correct / predictions.size() << endl;
// }


int main(int argc, char const *argv[]) {
    // assert(argc == 5);

    string dataset_path = "digits.txt";
    string classifier = "multinomial";
    double test_size = 0.5;
    double alpha = 1;

    auto dataset = read_dataset(dataset_path);
    auto data = dataset.first;
    auto labels = dataset.second;

    vector<vector<double> > train_data, test_data;
    vector<size_t> train_labels, test_labels;

    // split the dataset into training and testing sets
    for (size_t i = 0; i < data.size(); ++i) {
        if ((double)i / data.size() < test_size) {
            test_data.push_back(std::move(data[i]));
            test_labels.push_back(labels[i]);
        } else {
            train_data.push_back(std::move(data[i]));
            train_labels.push_back(labels[i]);
        }
    }
    
    if (classifier == "gaussian") {
        GaussianNB nb(alpha, {});
        nb.fit(train_data, train_labels);
        
        cout << "Accuracy: " << nb.score(test_data, test_labels) << endl;
    } else if (classifier == "multinomial") {
        MultinomialNB nb(alpha, {});

        
        nb.fit(train_data, train_labels);
        
        cout << "Accuracy: " << nb.score(test_data, test_labels) << endl;
    } else if (classifier == "bernoulli") {
        BernoulliNB nb(alpha, {});
        // binarize the data
        for (size_t i = 0; i < train_data.size(); ++i) {
            for (size_t j = 0; j < train_data[i].size(); ++j) {
                train_data[i][j] = (train_data[i][j] >= 8 ) ? 1 : 0;
            }
        }
        nb.fit(train_data, train_labels);
    
        cout << "Accuracy: " << nb.score(test_data, test_labels) << endl;
    } else {
        cout << "Invalid classifier type" << endl;
        exit(1);
    }


    return 0;
}