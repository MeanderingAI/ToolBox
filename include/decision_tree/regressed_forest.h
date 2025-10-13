#ifndef REGRESSION_FOREST_H
#define REGRESSION_FOREST_H

#include <random_forest.h>

class RegressionForest : public RandomForest
{
public:
	/**
	 * @brief Constructor for the RegressionForest.
	 * @param num_trees The number of regression trees in the forest.
	 * @param max_depth The maximum depth for each individual tree.
	 */
	RegressionForest(int num_trees, int max_depth);

	/**
	 * @brief Trains the regression forest using the provided dataset and weights.
	 * @param X The feature matrix.
	 * @param y The target vector of continuous values.
	 */
	void fit_weights(const std::vector<std::vector<int>> &X, const std::vector<int> &y);

	/**
	 * @brief Predicts the continuous value for a single sample.
	 * @param sample The feature vector for the sample.
	 * @return The predicted continuous value based on weighted average.
	 */
	int predict(const std::vector<int> &sample) const override;

private:
	std::vector<double> weights_;
}

#endif