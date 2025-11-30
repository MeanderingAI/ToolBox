#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

// Include all the ML headers
#include "decision_tree/decision_tree.h"
#include "decision_tree/random_forest.h"
// #include "decision_tree/decision_tree_regressor.h"  // TODO: Implement this class
#include "support_vector_machine/support_vector_machine.h"
#include "support_vector_machine/linear_kernel.h"
#include "support_vector_machine/rbf_kernel.h"
#include "support_vector_machine/polynomial_kernel.h"
#include "support_vector_machine/sigmoid_kernel.h"
#include "bayesian_network/bayesian_network.h"
#include "hidden_markov_model/hidden_markov_model.h"
#include "generalized_linear_model/linear_regression.h"
#include "multi_arm_bandit/bandit_arm.h"
#include "multi_arm_bandit/decaying_epsilon_agent.h"
#include "tracker/kalman_filter.h"
#include "tracker/unscented_kalman_filter.h"
#include "dimensionality_reduction/svd.h"
#include "dimensionality_reduction/pca.h"
#include "dimensionality_reduction/knn.h"
#include "dimensionality_reduction/umap.h"

namespace py = pybind11;

PYBIND11_MODULE(ml_core, m) {
    m.doc() = "Machine Learning Core Library Python Bindings";

    // Decision Tree Module
    py::module_ dt_module = m.def_submodule("decision_tree", "Decision Tree algorithms");
    
    // SplitCriterion enum
    py::enum_<SplitCriterion>(dt_module, "SplitCriterion")
        .value("GINI", SplitCriterion::GINI)
        .value("ENTROPY", SplitCriterion::ENTROPY);

    // DecisionTree class
    py::class_<DecisionTree>(dt_module, "DecisionTree")
        .def(py::init<SplitCriterion>(), py::arg("criterion") = SplitCriterion::GINI)
        .def("fit", &DecisionTree::fit,
             "Train the decision tree",
             py::arg("X"), py::arg("y"), py::arg("max_depth"))
        .def("predict", &DecisionTree::predict,
             "Predict class label for a sample",
             py::arg("sample"));

    // RandomForest class (assuming it exists)
    // py::class_<RandomForest>(dt_module, "RandomForest")
    //     .def(py::init<int, SplitCriterion>(), 
    //          py::arg("n_estimators") = 10, py::arg("criterion") = SplitCriterion::GINI)
    //     .def("fit", &RandomForest::fit)
    //     .def("predict", &RandomForest::predict);

    // Support Vector Machine Module
    py::module_ svm_module = m.def_submodule("svm", "Support Vector Machine algorithms");
    
    // Kernel base class
    py::class_<Kernel>(svm_module, "Kernel");
    
    // Linear Kernel
    py::class_<LinearKernel, Kernel>(svm_module, "LinearKernel")
        .def(py::init<>())
        .def("calculate", &LinearKernel::calculate);
    
    // RBF Kernel
    py::class_<RBFKernel, Kernel>(svm_module, "RBFKernel")
        .def(py::init<double>(), py::arg("gamma"))
        .def("calculate", &RBFKernel::calculate);
    
    // Polynomial Kernel
    py::class_<PolynomialKernel, Kernel>(svm_module, "PolynomialKernel")
        .def(py::init<int, double, double>(), 
             py::arg("degree"), py::arg("gamma"), py::arg("coef0"))
        .def("calculate", &PolynomialKernel::calculate);
    
    // Sigmoid Kernel
    py::class_<SigmoidKernel, Kernel>(svm_module, "SigmoidKernel")
        .def(py::init<double, double>(), py::arg("gamma"), py::arg("coef0"))
        .def("calculate", &SigmoidKernel::calculate);
    
    // SVM class
    py::class_<SVM>(svm_module, "SVM")
        .def(py::init<const Kernel&>(), py::arg("kernel"))
        .def("fit", &SVM::fit,
             "Train the SVM model",
             py::arg("X"), py::arg("y"))
        .def("predict", &SVM::predict,
             "Predict using the trained SVM",
             py::arg("sample"));

    // Bayesian Network Module
    py::module_ bn_module = m.def_submodule("bayesian_network", "Bayesian Network algorithms");
    
    // BayesianNetwork::Node struct
    py::class_<BayesianNetwork::Node>(bn_module, "Node")
        .def(py::init<>())
        .def_readwrite("name", &BayesianNetwork::Node::name)
        .def_readwrite("states", &BayesianNetwork::Node::states)
        .def_readwrite("index", &BayesianNetwork::Node::index);
    
    // BayesianNetwork class
    py::class_<BayesianNetwork>(bn_module, "BayesianNetwork")
        .def(py::init<>())
        .def("add_node", &BayesianNetwork::add_node,
             "Add a new node to the network",
             py::arg("node_name"), py::arg("states"))
        .def("add_edge", &BayesianNetwork::add_edge,
             "Add an edge between two nodes",
             py::arg("parent_index"), py::arg("child_index"))
        .def("set_cpt", &BayesianNetwork::set_cpt,
             "Set conditional probability table for a node",
             py::arg("node_index"), py::arg("cpt"))
        .def("calculate_joint_probability", &BayesianNetwork::calculate_joint_probability,
             "Calculate joint probability of an assignment",
             py::arg("assignment"))
        .def("infer", &BayesianNetwork::infer,
             "Perform probabilistic inference",
             py::arg("query_node_index"), py::arg("query_state_index"), py::arg("evidence"));

    // Hidden Markov Model Module
    py::module_ hmm_module = m.def_submodule("hmm", "Hidden Markov Model algorithms");
    
    py::class_<HMM>(hmm_module, "HMM")
        .def(py::init<int, int>(), py::arg("states"), py::arg("observations"))
        .def("set_initial_probabilities", &HMM::set_initial_probabilities,
             "Set initial state probabilities",
             py::arg("pi"))
        .def("set_transition_matrix", &HMM::set_transition_matrix,
             "Set state transition matrix",
             py::arg("A"))
        .def("set_emission_matrix", &HMM::set_emission_matrix,
             "Set observation emission matrix",
             py::arg("B"))
        .def("get_initial_probabilities", &HMM::get_initial_probabilities,
             "Get initial state probabilities")
        .def("get_transition_matrix", &HMM::get_transition_matrix,
             "Get state transition matrix")
        .def("get_emission_matrix", &HMM::get_emission_matrix,
             "Get observation emission matrix")
        .def("log_likelihood", &HMM::log_likelihood,
             "Calculate log likelihood of observation sequence",
             py::arg("observations"))
        .def("get_most_likely_states", &HMM::get_most_likely_states,
             "Get most likely hidden state sequence (Viterbi algorithm)",
             py::arg("observations"))
        .def("train", &HMM::train,
             "Train HMM using Baum-Welch algorithm",
             py::arg("observation_sequences"), 
             py::arg("max_iterations") = 100,
             py::arg("tolerance") = 1e-6,
             py::arg("smoothing_factor") = 0,
             py::arg("seed") = 0);

    // Generalized Linear Model Module
    py::module_ glm_module = m.def_submodule("glm", "Generalized Linear Model algorithms");
    
    // FitMethod base class
    py::class_<FitMethod>(glm_module, "FitMethod");
    
    // LinearRegressionFitMethod
    py::enum_<LinearRegressionFitMethod::Type>(glm_module, "LinearRegressionType")
        .value("GRADIENT_DESCENT", LinearRegressionFitMethod::Type::GRADIENT_DESCENT)
        .value("CLOSED_FORM", LinearRegressionFitMethod::Type::CLOSED_FORM);
    
    py::class_<LinearRegressionFitMethod, FitMethod>(glm_module, "LinearRegressionFitMethod")
        .def(py::init<unsigned int, double, LinearRegressionFitMethod::Type>(),
             py::arg("num_iterations") = 10000,
             py::arg("learning_rate") = 0.01,
             py::arg("type") = LinearRegressionFitMethod::Type::GRADIENT_DESCENT)
        .def("get_num_iterations", &LinearRegressionFitMethod::get_num_iterations)
        .def("get_learning_rate", &LinearRegressionFitMethod::get_learning_rate)
        .def("get_type", &LinearRegressionFitMethod::get_type);
    
    // GLM base class
    py::class_<GLM>(glm_module, "GLM")
        .def("fit", &GLM::fit,
             "Train the model",
             py::arg("X"), py::arg("y"))
        .def("predict", &GLM::predict,
             "Predict output for a sample",
             py::arg("sample"));
    
    // LinearRegression class
    py::class_<LinearRegression, GLM>(glm_module, "LinearRegression")
        .def(py::init<const LinearRegressionFitMethod&>(), py::arg("fit_method"))
        .def("fit", &LinearRegression::fit,
             "Train the linear regression model",
             py::arg("X"), py::arg("y"))
        .def("predict", &LinearRegression::predict,
             "Predict using the trained model",
             py::arg("sample"))
        .def("get_coefficients", &LinearRegression::get_coefficients,
             "Get learned coefficients (weights and bias)");

    // Multi-arm Bandit Module
    py::module_ mab_module = m.def_submodule("multi_arm_bandit", "Multi-arm Bandit algorithms");
    
    py::class_<BanditArm>(mab_module, "BanditArm")
        .def(py::init<double>(), py::arg("true_reward_prob"))
        .def("pull", &BanditArm::pull,
             "Pull the bandit arm and get reward")
        .def("update", &BanditArm::update,
             "Update arm statistics with reward",
             py::arg("reward"))
        .def("get_estimated_prob", &BanditArm::get_estimated_prob,
             "Get estimated reward probability")
        .def("get_pull_count", &BanditArm::get_pull_count,
             "Get number of times arm was pulled")
        .def("get_true_prob", &BanditArm::get_true_prob,
             "Get true reward probability");

    // Tracker Module (Kalman Filters)
    py::module_ tracker_module = m.def_submodule("tracker", "State estimation and tracking algorithms");
    
    // Note: Kalman Filter bindings would need to be implemented based on the actual interface
    // This is a placeholder - you'll need to check the actual KalmanFilter class interface
    /*
    py::class_<KalmanFilter>(tracker_module, "KalmanFilter")
        .def(py::init<>())
        .def("predict", &KalmanFilter::predict)
        .def("update", &KalmanFilter::update);
    */

    // Dimensionality Reduction Module
    py::module_ dr_module = m.def_submodule("dimensionality_reduction", 
                                             "Dimensionality reduction algorithms (SVD, PCA)");
    
    using namespace dimensionality_reduction;
    
    // SVD class
    py::class_<SVD>(dr_module, "SVD")
        .def(py::init<bool>(), py::arg("compute_full_matrices") = false,
             "Create SVD object\n\n"
             "Args:\n"
             "    compute_full_matrices: If True, compute full U and V matrices")
        .def("compute", &SVD::compute,
             "Compute SVD of matrix X\n\n"
             "Args:\n"
             "    X: Input matrix (m x n)",
             py::arg("X"))
        .def("get_U", &SVD::get_U,
             "Get left singular vectors (U matrix)\n\n"
             "Returns:\n"
             "    U matrix (m x k) where k = min(m,n) for thin SVD")
        .def("get_singular_values", &SVD::get_singular_values,
             "Get singular values as vector\n\n"
             "Returns:\n"
             "    Vector of singular values in descending order")
        .def("get_V", &SVD::get_V,
             "Get right singular vectors (V matrix)\n\n"
             "Returns:\n"
             "    V matrix (n x k) where k = min(m,n) for thin SVD")
        .def("get_S", &SVD::get_S,
             "Get singular values as diagonal matrix\n\n"
             "Returns:\n"
             "    Diagonal matrix of singular values")
        .def("reconstruct", &SVD::reconstruct,
             "Reconstruct matrix from SVD components\n\n"
             "Args:\n"
             "    num_components: Number of components to use (0 = all)\n\n"
             "Returns:\n"
             "    Reconstructed matrix",
             py::arg("num_components") = 0)
        .def("rank", &SVD::rank,
             "Get rank of matrix\n\n"
             "Args:\n"
             "    tolerance: Tolerance for considering singular values as zero\n\n"
             "Returns:\n"
             "    Estimated rank",
             py::arg("tolerance") = -1.0)
        .def("explained_variance_ratio", &SVD::explained_variance_ratio,
             "Get explained variance ratio for each component\n\n"
             "Returns:\n"
             "    Vector of explained variance ratios")
        .def("is_computed", &SVD::is_computed,
             "Check if SVD has been computed\n\n"
             "Returns:\n"
             "    True if compute() has been called");
    
    // PCA class
    py::class_<PCA>(dr_module, "PCA")
        .def(py::init<int, bool, bool>(),
             py::arg("n_components") = 0,
             py::arg("center") = true,
             py::arg("scale") = false,
             "Create PCA object\n\n"
             "Args:\n"
             "    n_components: Number of components to keep (0 = keep all)\n"
             "    center: If True, center data by subtracting mean\n"
             "    scale: If True, scale data to unit variance")
        .def("fit", &PCA::fit,
             "Fit PCA to data matrix X\n\n"
             "Args:\n"
             "    X: Data matrix (rows = samples, cols = features)",
             py::arg("X"))
        .def("transform", &PCA::transform,
             "Transform data to principal component space\n\n"
             "Args:\n"
             "    X: Data matrix to transform\n\n"
             "Returns:\n"
             "    Transformed data (rows = samples, cols = n_components)",
             py::arg("X"))
        .def("fit_transform", &PCA::fit_transform,
             "Fit and transform in one step\n\n"
             "Args:\n"
             "    X: Data matrix\n\n"
             "Returns:\n"
             "    Transformed data",
             py::arg("X"))
        .def("inverse_transform", &PCA::inverse_transform,
             "Inverse transform from PC space back to original space\n\n"
             "Args:\n"
             "    X_transformed: Data in PC space\n\n"
             "Returns:\n"
             "    Reconstructed data in original space",
             py::arg("X_transformed"))
        .def("get_components", &PCA::get_components,
             "Get principal components (loadings)\n\n"
             "Returns:\n"
             "    Matrix where each column is a principal component")
        .def("get_explained_variance", &PCA::get_explained_variance,
             "Get explained variance for each component\n\n"
             "Returns:\n"
             "    Vector of explained variances")
        .def("get_explained_variance_ratio", &PCA::get_explained_variance_ratio,
             "Get explained variance ratio for each component\n\n"
             "Returns:\n"
             "    Vector of explained variance ratios (sum to 1.0)")
        .def("get_singular_values", &PCA::get_singular_values,
             "Get singular values\n\n"
             "Returns:\n"
             "    Vector of singular values")
        .def("get_mean", &PCA::get_mean,
             "Get mean of training data\n\n"
             "Returns:\n"
             "    Mean vector")
        .def("get_scale", &PCA::get_scale,
             "Get standard deviation of training data\n\n"
             "Returns:\n"
             "    Standard deviation vector")
        .def("get_n_components", &PCA::get_n_components,
             "Get number of components\n\n"
             "Returns:\n"
             "    Number of components kept")
        .def("is_fitted", &PCA::is_fitted,
             "Check if PCA has been fitted\n\n"
             "Returns:\n"
             "    True if fit() has been called");
    
    // KNN class
    py::class_<KNN>(dr_module, "KNN")
        .def(py::init<int, const std::string&>(),
             py::arg("k") = 5,
             py::arg("metric") = "euclidean",
             "Create KNN object\n\n"
             "Args:\n"
             "    k: Number of nearest neighbors\n"
             "    metric: Distance metric ('euclidean', 'manhattan', 'cosine')")
        .def("fit", &KNN::fit,
             "Fit KNN with training data\n\n"
             "Args:\n"
             "    X: Training data matrix",
             py::arg("X"))
        .def("kneighbors", 
             py::overload_cast<const Eigen::MatrixXd&>(&KNN::kneighbors, py::const_),
             "Find k-nearest neighbors for query points\n\n"
             "Args:\n"
             "    X_query: Query points\n\n"
             "Returns:\n"
             "    Tuple of (indices, distances) matrices",
             py::arg("X_query"))
        .def("kneighbors",
             py::overload_cast<>(&KNN::kneighbors, py::const_),
             "Find k-nearest neighbors for training data\n\n"
             "Returns:\n"
             "    Tuple of (indices, distances) matrices")
        .def("pairwise_distances", &KNN::pairwise_distances,
             "Compute pairwise distances\n\n"
             "Args:\n"
             "    X: First set of points\n"
             "    Y: Second set of points\n\n"
             "Returns:\n"
             "    Distance matrix",
             py::arg("X"), py::arg("Y"))
        .def("get_k", &KNN::get_k,
             "Get number of neighbors")
        .def("get_metric", &KNN::get_metric,
             "Get distance metric")
        .def("is_fitted", &KNN::is_fitted,
             "Check if KNN has been fitted");
    
    // UMAP class
    py::class_<UMAP>(dr_module, "UMAP")
        .def(py::init<int, int, double, const std::string&, double, int, int>(),
             py::arg("n_components") = 2,
             py::arg("n_neighbors") = 15,
             py::arg("min_dist") = 0.1,
             py::arg("metric") = "euclidean",
             py::arg("learning_rate") = 1.0,
             py::arg("n_epochs") = 200,
             py::arg("random_state") = 42,
             "Create UMAP object\n\n"
             "Args:\n"
             "    n_components: Number of dimensions in embedding\n"
             "    n_neighbors: Number of nearest neighbors\n"
             "    min_dist: Minimum distance between embedded points\n"
             "    metric: Distance metric ('euclidean', 'manhattan', 'cosine')\n"
             "    learning_rate: Learning rate for optimization\n"
             "    n_epochs: Number of optimization epochs\n"
             "    random_state: Random seed for reproducibility")
        .def("fit", &UMAP::fit,
             "Fit UMAP to data\n\n"
             "Args:\n"
             "    X: Input data matrix",
             py::arg("X"))
        .def("transform", &UMAP::transform,
             "Transform data to embedding\n\n"
             "Args:\n"
             "    X: Data to transform\n\n"
             "Returns:\n"
             "    Embedded data",
             py::arg("X"))
        .def("fit_transform", &UMAP::fit_transform,
             "Fit and transform in one step\n\n"
             "Args:\n"
             "    X: Input data\n\n"
             "Returns:\n"
             "    Embedded data",
             py::arg("X"))
        .def("get_embedding", &UMAP::get_embedding,
             "Get the learned embedding\n\n"
             "Returns:\n"
             "    Embedding matrix")
        .def("is_fitted", &UMAP::is_fitted,
             "Check if UMAP has been fitted")
        .def("get_n_components", &UMAP::get_n_components,
             "Get number of components")
        .def("get_n_neighbors", &UMAP::get_n_neighbors,
             "Get number of neighbors");
}