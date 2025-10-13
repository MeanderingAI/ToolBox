#ifndef ML_DATA_H
#define ML_DATA_H

#include <vector>
#include <map>
#include <string>
#include <variant>
#include <memory>
#include <Eigen/Dense>

namespace MLBase {

// Enum to specify the data type
enum class DataType {
    INTEGER,
    DOUBLE,
    STRING,
    DISCRETE_STATE,
    EIGEN_MATRIX,
    EIGEN_VECTOR
};

// Variant type that can hold different data formats
using DataValue = std::variant<
    int,
    double,
    std::string,
    std::vector<int>,
    std::vector<double>,
    std::vector<std::string>,
    std::vector<std::vector<int>>,
    std::vector<std::vector<double>>,
    std::map<int, int>,
    std::map<std::string, std::string>,
    Eigen::MatrixXd,
    Eigen::VectorXd
>;

/**
 * @class MLData
 * @brief A unified data container for all machine learning algorithms
 * 
 * This class provides a flexible interface to handle different data types
 * used across various machine learning algorithms in the codebase.
 */
class MLData {
public:
    // Constructors for different data types
    MLData();
    MLData(const std::vector<std::vector<int>>& features, const std::vector<int>& targets);
    MLData(const std::vector<std::vector<double>>& features, const std::vector<double>& targets);
    MLData(const Eigen::MatrixXd& features, const Eigen::VectorXd& targets);
    MLData(const std::vector<int>& sequence_data);
    MLData(const std::vector<std::vector<int>>& sequence_data);
    MLData(const std::map<int, int>& discrete_assignments);

    // Getters for different data formats
    std::vector<std::vector<int>> get_int_features() const;
    std::vector<int> get_int_targets() const;
    std::vector<std::vector<double>> get_double_features() const;
    std::vector<double> get_double_targets() const;
    Eigen::MatrixXd get_eigen_features() const;
    Eigen::VectorXd get_eigen_targets() const;
    std::vector<int> get_sequence() const;
    std::vector<std::vector<int>> get_sequences() const;
    std::map<int, int> get_discrete_assignments() const;

    // Utility methods
    size_t get_num_samples() const;
    size_t get_num_features() const;
    DataType get_primary_type() const;
    bool is_valid() const;

    // Conversion methods
    void convert_to_eigen();
    void convert_to_double_vectors();
    void convert_to_int_vectors();

    // Metadata
    void set_feature_names(const std::vector<std::string>& names);
    std::vector<std::string> get_feature_names() const;

private:
    DataValue features_;
    DataValue targets_;
    DataType primary_type_;
    std::vector<std::string> feature_names_;
    bool valid_;

    void validate_data();
};

} // namespace MLBase

#endif // ML_DATA_H