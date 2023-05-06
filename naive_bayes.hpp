#ifndef NAIVE_BAYES_HPP
#define NAIVE_BAYES_HPP

#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <array>
#include <iterator>
#include <cmath>
#include <algorithm>


// the base clas for naive bayes
class NaiveBayes {
public:
    void fit(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have the same number of samples");
        }

        if (!set_priors) {
            n_classes = *std::max_element(y.begin(), y.end()) + 1;
            class_priors = std::vector<double>(n_classes, 0);
            compute_priors(y);
        } else {
            n_classes = class_priors.size();
        }
    }

    std::vector<double> predict_single_proba(const std::vector<double> &x);

    size_t predict_single(const std::vector<double> &x) {
        auto probabilities = predict_single_proba(x);
        return argmax(probabilities.begin(), probabilities.end());
    }

    std::vector<size_t> predict(const std::vector<std::vector<double> > &X) {
        std::vector<size_t> predictions(X.size(), 0);
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] = predict_single(X[i]);
        }

        return predictions;
    }

    std::vector<std::vector<double> > predict_proba(const std::vector<std::vector<double> > &X) {
        std::vector<std::vector<double> > probabilities(X.size(), std::vector<double>(n_classes, 0));
        for (size_t i = 0; i < X.size(); ++i) {
            probabilities[i] = predict_single_proba(X[i]);
        }

        return probabilities;
    }

    double score(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have the same number of samples");
        }

        size_t correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            if (predict_single(X[i]) == y[i]) {
                correct += 1;
            }
        }

        return (double)correct / X.size();
    }

    template<typename Iter>
    size_t argmax(const Iter &begin, const Iter &end) {
        return std::distance(begin, std::max_element(begin, end));
    }
protected:
    void compute_priors(const std::vector<size_t> &y) {
        for (size_t i = 0; i < y.size(); ++i) {
            class_priors[y[i]] += 1;
        }

        for (size_t i = 0; i < n_classes; ++i) {
            class_priors[i] /= y.size();
        }
    }

    bool set_priors;
    std::vector<double> class_priors;
    size_t n_classes;
};


class GaussianNB : public NaiveBayes {
public:
    GaussianNB(double smoothing=1e-9, const std::vector<double> &class_priors={}) {
        set_priors = !class_priors.empty();
        this->class_priors = class_priors;
        this->smoothing = smoothing;
    }

    void fit(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        NaiveBayes::fit(X, y);

        // initialize the means and variances
        means = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        variances = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        class_counts = std::vector<size_t>(n_classes, 0);

        compute_means(X, y);
        compute_variances(X, y);
    }

    std::vector<double> predict_single_proba(const std::vector<double> &x) {
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

    double normal_pdf(double x, double mean, double variance) {
        return 1 / std::sqrt(2 * M_PI * variance) * std::exp(- (x - mean) * (x - mean) / (2 * variance));
    }

    double get_smoothing() const {
        return smoothing;
    }

    void set_smoothing(double smoothing) {
        this->smoothing = smoothing;
    }

    std::vector<double> get_priors() const {
        return class_priors;
    }

    // void set_priors(std::vector<double> priors) {
    //     this->priors = priors;
    //     set_priors = true;
    // }

    ~GaussianNB() = default;
private:
    void compute_means(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
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

    void compute_variances(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                variances[y[i]][j] += (X[i][j] - means[y[i]][j]) * (X[i][j] - means[y[i]][j]);
            }
        }

        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                variances[y[i]][j] = std::sqrt(variances[y[i]][j] / class_counts[y[i]] + smoothing);
            }
        }
    }

    double smoothing;
    std::vector<std::vector<double> > means;
    std::vector<std::vector<double> > variances;
    std::vector<size_t> class_counts;
};


class BernoulliNB : public NaiveBayes {
public:
    BernoulliNB(double alpha=1.0, const std::vector<double> &class_priors={}) {
        set_priors = !class_priors.empty();
        this->class_priors = class_priors; 
        this->alpha = alpha;
    }

    void fit(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        NaiveBayes::fit(X, y);

        auto means = compute_means(X, y);
        compute_biases(X, y, means);
        compute_weights(means);
    }

    std::vector<double> predict_single_proba(const std::vector<double> &x) {
        std::vector<double> probabilities(n_classes, 0);
        for (size_t i = 0; i < n_classes; ++i) {
            probabilities[i] = biases[i];
            for (size_t j = 0; j < x.size(); ++j) {
                probabilities[i] += x[j] * weights[i][j];
            }
        }

        return probabilities;
    }

    double get_alpha() const {
        return alpha;
    }

    void set_alpha(double alpha) {
        this->alpha = alpha;
    }

    std::vector<double> get_priors() const {
        return class_priors;
    }

    // void set_priors(std::vector<double> priors) {
    //     this->priors = priors;
    //     set_priors = true;
    // }
private:
    std::vector<std::vector<double> > compute_means(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        auto means = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        class_counts = std::vector<size_t>(n_classes, 0);

        // first a pass to compute the sums and class counts
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                means[y[i]][j] += X[i][j];
            }
            class_counts[y[i]] += 1;
        }

        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                means[y[i]][j] = (means[y[i]][j] + alpha) / (class_counts[y[i]] + 2 * alpha);
            }
        }

        return means;
    }

    void compute_biases(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y, const std::vector<std::vector<double> > &means) {
        biases = std::vector<double>(n_classes, 0);

        for (size_t k = 0; k < n_classes; ++k) {
            biases[k] = std::log(class_priors[k]);
            for (size_t i = 0; i < X.size(); ++i) {
                biases[k] += std::log(1 - means[k][i]);
            }
        }
    }

    void compute_weights(const std::vector<std::vector<double> > &means) {
        for (size_t i = 0; i < means.size(); ++i) {
            for (size_t j = 0; j < means[0].size(); ++j) {
                weights[i][j] = std::log(means[i][j] / (1 - means[i][j]));
            }
        }
    }

    std::vector<size_t> class_counts;
    std::vector<double> biases;
    std::vector<std::vector<double> > weights;
    double alpha;
};


class MultinomialNB : public NaiveBayes {
public:
    MultinomialNB(double alpha=1.0, const std::vector<double> class_priors={}) {
        set_priors = !class_priors.empty();
        this->class_priors = class_priors;
        this->alpha = alpha;
    }

    void fit(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        NaiveBayes::fit(X, y);

        compute_weights(X, y);
    }

    std::vector<double> predict_single_proba(const std::vector<double> &x) {
        std::vector<double> logits(n_classes, 0);

        for (size_t i = 0; i < n_classes; ++i) {
            logits[i] = class_priors[i];
            for (size_t j = 0; j < x.size(); ++j) {
                logits[i] += x[j] * weights[i][j];
            }
        }

        return logits;
    }

private:
    /**
     * Compute the sum of the features for each class
     * @param X the data
     * @param y the labels
     * @return the sum of the features for each class
    */
    std::vector<std::vector<double> > compute_class_features_sum(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        std::vector<std::vector<double> > class_features_sum(n_classes, std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                class_features_sum[y[i]][j] += X[i][j];
            }
        }

        return class_features_sum;
    }

    /**
     * Compute the sum of features across all classes
     * @return the sum of features across all classes
    */
    std::vector<double> compute_sums_across_all_classes(const std::vector<std::vector<double> > &class_features_sum) {
        const size_t D = class_features_sum[0].size();
        std::vector<double> sums(n_classes, 0);

        for (size_t i = 0; i < n_classes; ++i) {
            for (size_t j = 0; j < D; ++j) {
                sums[i] += class_features_sum[i][j];
            }
        }

        return sums;
    }

    void compute_weights(const std::vector<std::vector<double> > &X, const std::vector<size_t> &y) {
        auto class_features_sum = compute_class_features_sum(X, y);
        auto sums = compute_sums_across_all_classes(class_features_sum);

        weights = std::vector<std::vector<double> >(n_classes, std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < n_classes; ++i) {
            for (size_t j = 0; j < X[0].size(); ++j) {
                weights[i][j] = std::log((class_features_sum[i][j] + alpha) / (sums[i] + alpha * X[0].size()));
            }
        }
    }

    double alpha;
    bool set_priors;
    size_t n_classes;
    std::vector<std::vector<double> > weights;
};


#endif  // NAIVE_BAYES_HPP