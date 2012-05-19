#ifndef RANDOM_FOREST_COMMON_H
#define RANDOM_FOREST_COMMON_H

#include <vector>
#include <string>
#include <sstream>

#include <Eigen/Core>

//#ifndef EIGEN_CORE_H
//namespace Eigen {
//  class MatrixXf;
//  class RowVectorXf;
//}
//#endif

namespace RandomForest {

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;


int CountCommas(const std::string& s);

template<typename T>
std::vector<T> StringToStdVector(const std::string& line, int n_elements = 0) {
  // if number of elements was not specified, count the commas
  if (n_elements == 0) { n_elements = CountCommas(line) + 1; }
  // fill in the vector
  std::vector<T> vec(n_elements);
  std::stringstream ss(line);
	std::string value;
	for (int el = 0; el < n_elements; ++el) {
		getline(ss, value, ',');
		istringstream(value) >> vec[el];
	}
	return vec;
};

template<typename T>
std::string StdVectorToString(const std::vector<T>& vec) {
  std::ostringstream oss;
  std::vector<T>::const_iterator cit = vec.begin();
  oss << *cit;
  for (; cit != vec.end(); ++cit) {
    oss << "," << *cit;
  }
  return oss.str();
};

RowVector ReadFloatVector(const std::string& line, int size);
Matrix ReadFloatMatrix(const std::string& line, int height, int width);

} // namespace RandomForest

#endif // RANDOM_FOREST_COMMON_H