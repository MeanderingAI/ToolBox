
#include <regression_forest.h>

RegressionForest::RegressionForest(int num_trees, int max_depth)
	: RandomForest(num_trees, max_depth)
{
	weights_.resize(num_trees, 1.0 / num_trees); // Initialize equal weights
}

void RegressionForest::fit_weights(const std::vector<std::vector<int>> &X, const std::vector<int> &y)
{
	if (X.empty() || y.empty() || X.size() != y.size())
	{
		return; // Handle invalid input
	}

	// 1. Calculate the Mean Squared Error (MSE) for each tree.
	std::vector<double> mse_per_tree(num_trees_, 0.0);
	for (size_t i = 0; i < trees_.size(); ++i)
	{
		double squared_error_sum = 0.0;
		for (size_t j = 0; j < X.size(); ++j)
		{
			double prediction = trees_[i]->predict(X[j]);
			squared_error_sum += std::pow(prediction - y[j], 2);
		}
		mse_per_tree[i] = squared_error_sum / X.size();
	}

	// 2. Compute weights based on the inverse of the MSE.
	// A small epsilon is added to avoid division by zero.
	double total_inverse_mse = 0.0;
	double epsilon = 1e-9;
	for (size_t i = 0; i < num_trees_; ++i)
	{
		double weight = 1.0 / (mse_per_tree[i] + epsilon);
		weights_[i] = weight;
		total_inverse_mse += weight;
	}

	// 3. Normalize the weights so they sum to 1.
	if (total_inverse_mse > 0)
	{
		for (size_t i = 0; i < num_trees_; ++i)
		{
			weights_[i] /= total_inverse_mse;
		}
	}
}

int RegressionForest::predict(const std::vector<int> &sample) const
{
	double weighted_sum = 0.0;
	for (size_t i = 0; i < trees_.size(); ++i)
	{
		weighted_sum += trees_[i]->predict(sample) * weights_[i];
	}
	return weighted_sum;
}