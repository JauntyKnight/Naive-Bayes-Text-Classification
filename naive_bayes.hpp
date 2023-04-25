#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <array>
#include <iterator>
#include <cmath>


// the base clas for naive bayes
class NaiveBayes {
public:
    virtual void fit(std::vector<std::vector<double> > X, std::vector<size_t> y);
    virtual std::vector<size_t> predict(std::vector<std::vector<double> > X);
    virtual double score(std::vector<std::vector<double> > X, std::vector<size_t> y);
    virtual std::vector<std::vector<double> > predict_proba(std::vector<std::vector<double> > X);
    virtual std::vector<double> predict_single_proba(std::vector<double> x);
    virtual size_t predict_single(std::vector<double> x);

    virtual ~NaiveBayes() = default;

    template<typename Iter>
    size_t argmax(const Iter &begin, const Iter &end) {
        return std::distance(begin, std::max_element(begin, end));
    }
};


class GaussianNB : public NaiveBayes {
    GaussianNB(double smoothing=1e-9) : smoothing(smoothing) {}

    void fit(std::vector<std::vector<double> > X, std::vector<size_t> y) {
        n_classes = *std::max_element(y.begin(), y.end()) + 1;

        // initialize the means and variances
        means = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        variances = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        class_priors = std::vector<double>(n_classes, 0);
        class_counts = std::vector<size_t>(n_classes, 0);

        compute_means(X, y);
        compute_variances(X, y);

        // compute the class priors
        for (size_t i = 0; i < y.size(); ++i) {
            class_priors[y[i]] += 1;
        }

        for (size_t i = 0; i < n_classes; ++i) {
            class_priors[i] /= y.size();
        }
    }

    std::vector<double> predict_single_proba(std::vector<double> x) {
        std::vector<double> probabilities;
        for (size_t i = 0; i < n_classes; ++i) {
            double probability = class_priors[i];
            for (size_t j = 0; j < x.size(); ++j) {
                probability *= normal_pdf(x[j], means[i][j], variances[i][j]);
            }
            probabilities.push_back(probability);
        }

        return probabilities;
    }

    size_t predict_single(std::vector<double> x) {
        auto probabilities = predict_single_proba(x);
        return argmax(probabilities.begin(), probabilities.end());
    }

    std::vector<std::vector<double> > predict_proba(std::vector<std::vector<double> > X) {
        std::vector<std::vector<double> > probabilities;
        for (size_t i = 0; i < X.size(); ++i) {
            probabilities.push_back(predict_single_proba(X[i]));
        }

        return probabilities;
    }

    std::vector<size_t> predict(std::vector<std::vector<double> > X) {
        std::vector<size_t> predictions;
        for (size_t i = 0; i < X.size(); ++i) {
            predictions.push_back(predict_single(X[i]));
        }

        return predictions;
    }

    double score(std::vector<std::vector<double> > X, std::vector<size_t> y) {
        auto predictions = predict(X);
        size_t correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == y[i]) {
                correct += 1;
            }
        }

        return (double)correct / predictions.size();
    }
private:
    void compute_means(std::vector<std::vector<double> > X, std::vector<size_t> y) {
        // first a pass to compute the sums and class counts
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                means[y[i]][j] += X[i][j];
            }
            class_counts[y[i]] += 1;
        }

        // the second pass to compute the means
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                means[y[i]][j] /= class_counts[y[i]];
            }
        }
    }

    void compute_variances(std::vector<std::vector<double> > X, std::vector<size_t> y) {
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                variances[y[i]][j] += (X[i][j] - means[y[i]][j]) * (X[i][j] - means[y[i]][j]);
            }
        }

        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                variances[y[i]][j] = std::sqrt(variances[y[i]][j] / class_counts[y[i]] + smoothing)
            }
        }
    }

    double normal_pdf(double x, double mean, double variance) {
        return 1 / std::sqrt(2 * std::M_PI * variance) * std::exp(- (x - mean) * (x - mean) / (2 * variance));
    }

    double smoothing;
    size_t n_classes;
    std::vector<std::vector<double> > means;
    std::vector<std::vector<double> > variances;
    std::vector<size_t> class_counts;
    std::vector<double> class_priors;
};
