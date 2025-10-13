#ifndef DECISION_TREE_REGRESSOR_H
#define DECISION_TREE_REGRESSOR_H

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <numeric> // For std::accumulate

// Represents a single node in the regression tree.
struct Node {
    // A flag to check if the node is a leaf node.
    bool is_leaf = false;
    // The predicted value if it is a leaf node (the mean of the target values).
    double prediction_value = 0.0; 
    // The index of the feature to split on at this node.
    int feature_index = -1;
    // A map from a feature value to a child node.
    std::map<int, Node*> children;
};

// Helper function declarations
double calculate_variance(const std::vector<double>& y);
double calculate_mean(const std::vector<double>& y);

// Represents a Decision Tree Regressor.
class DecisionTreeRegressor {
public:
    // Constructor.
    DecisionTreeRegressor(int min_samples_split = 2);
    
    // Destructor to clean up memory.
    ~DecisionTreeRegressor();

    /**
     * @brief Trains the decision tree regressor using the provided dataset.
     * @param X The feature matrix (assumed discrete/ordinal).
     * @param y The target vector of continuous values (double).
     * @param max_depth The maximum depth of the tree to prevent overfitting.
     */
    void fit(const std::vector<std::vector<int>>& X, const std::vector<double>& y, int max_depth);

    /**
     * @brief Predicts the target value for a single sample.
     * @param sample The feature vector for the sample to predict.
     * @return The predicted target value (double).
     */
    double predict(const std::vector<int>& sample) const;

private:
    Node* root;
    int max_depth;
    int min_samples_split_;

    // Helper function to recursively build the tree.
    Node* build_tree_recursive(const std::vector<std::vector<int>>& X, const std::vector<double>& y, int current_depth);
    
    // Helper function to find the best feature to split on using variance reduction.
    int find_best_split(
        const std::vector<std::vector<int>>& X, 
        const std::vector<double>& y,
        std::map<int, std::vector<double>>& best_split_groups
    );

    // Helper function to delete the tree.
    void delete_tree_recursive(Node* node);

    // Friend declarations for utility functions.
    friend double calculate_variance(const std::vector<double>& y);
    friend double calculate_mean(const std::vector<double>& y);
};

#endif // DECISION_TREE_REGRESSOR_H