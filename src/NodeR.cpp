#include "NodeR.h"

#include <iostream>
#include <fstream>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::vector;
using namespace RandomForest;

NodeR::NodeR(
    int id,
    const Matrix& data,
    const Matrix& labels,
    const vector<int>& samples):

	  id_(id),
    n_samples_(samples.size()),
    samples_(samples) {
	InitNode(data, labels);
}

NodeR::NodeR(
    int id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity):

	  id_(id),
	  is_leaf_(is_leaf),
	  split_dim_(split_dim),
	  split_thresh_(split_thresh),
    impurity_(impurity) {
  assert(!is_leaf);
}

NodeR::NodeR(int id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity,
    const NormalDistribution& nd,
    const vector<int>& samples):

	  id_(id),
	  is_leaf_(is_leaf),
	  split_dim_(split_dim),
	  split_thresh_(split_thresh),
    impurity_(impurity),
	  nd_(nd),
    n_samples_(samples.size()),
	  samples_(samples) {
	assert(is_leaf);
}

NodeR::~NodeR(void) {	
	if (!is_leaf_) {
		delete left_;
		delete right_;
	}
}

void NodeR::InitNode(const Matrix& data, const Matrix& labels) {
	CalcDimLimits(data);
	//labels.med
	nd_ = getSampleDistr(labels, samples_);
  //// DBG
  //if (id_ == 31) {
  //  std::cout << std::endl;
  //  std::cout << "creating node " << id_ << ":" << std::endl;
  //  std::cout << "data: " << SubsetRowwise(labels, samples_) << std::endl;
  //  std::cout << nd_.mean() << std::endl;
  //  std::cout << nd_.covariance() << std::endl;
  //  std::string s;
  //  std::getline(std::cin, s);
  //}

	impurity_ = DistributionImpurity(nd_);
	is_leaf_ = false;
}

void NodeR::CalcDimLimits(const Matrix& data) {
	assert(samples_.size() > 0);

	dim_mins_ = vector<double>(data.cols());
	dim_maxes_ = vector<double>(data.cols());

	for (int col = 0; col < data.cols(); col++) {
		dim_mins_[col] = DBL_MAX;
		dim_maxes_[col] = DBL_MIN;
	}

	for (int col = 0; col < data.cols(); ++col) {
    dim_mins_[col] = data.col(col).minCoeff();
    dim_maxes_[col] = data.col(col).maxCoeff();
	}
}

NormalDistribution NodeR::CalcFeatureDist(const Matrix& data) {
	return getSampleDistr(data, samples_);
}

NodeR* NodeR::GetTerminalNode(const RowVector& features) {
  if (is_leaf_) { return this; }
  return features(split_dim_) < split_thresh_ ? left_->GetTerminalNode(features) : right_->GetTerminalNode(features);
}

NormalDistribution NodeR::EvaluatePoint(const RowVector& features) {
  return GetTerminalNode(features)->nd_;
}

float NodeR::GetFeatureDistanceMahalanobis(
    const RowVector& featurePoint,
    const Matrix& data) {
  NormalDistribution featureDistribution = getSampleDistr(data, samples_);
  RowVector difference = featurePoint - featureDistribution.mean();
  Matrix diagonalCov = featureDistribution.covariance().diagonal().asDiagonal();

  float dist = (difference * diagonalCov.inverse() * difference.transpose())[0];

  return dist;
}

void NodeR::WriteNode(std::ofstream& file) const {
	file << id_ << std::endl;
	file << is_leaf_ << std::endl;
  file << n_samples_ << std::endl;
	file << split_dim_ << std::endl;
	file << split_thresh_ << std::endl;
  file << impurity_ << std::endl;
	
	if (is_leaf_) {
    //// DBG
    //if (id_ == 31) {
    //  std::cout << std::endl;
    //  std::cout << "saving node " << id_ << ":" << std::endl;
    //  std::cout << std::endl << nd_.mean() << std::endl;
    //  std::cout << std::endl << nd_.covariance() << std::endl;
    //  std::string s;
    //  std::getline(std::cin, s);
    //}
    file << matrixString(nd_.mean()) << std::endl;
	  file << matrixString(nd_.covariance()) << std::endl;
    file << intVectorString(samples_) << std::endl;
	} else {
    file << RandomForest::StdVectorToString(info_gains_) << std::endl;
		left_->WriteNode(file);
		right_->WriteNode(file);
	}
}

std::string NodeR::matrixString(const Matrix& mat) {
	std::ostringstream oss;
	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			oss << mat(r, c);
			if ((r+1)*(c+1) < mat.rows() * mat.cols()) {
				oss << ",";
			}
		}
	}
	return oss.str();
}

std::string NodeR::intVectorString(const vector<int> vec) {
  std::ostringstream oss;
  oss << vec[0];
  for (int el = 1; el < int(vec.size()); ++el) {
    oss << "," << vec[el];
  }
  return oss.str();
}

NormalDistribution RandomForest::getSampleDistr(
    const Matrix& data,
    const vector<int>& samples) {
	int nDimOut = data.cols();
	int nSamples(samples.size());

	Matrix l(int(samples.size()), data.cols());
	for (int r = 0; r < l.rows(); r++) {
    l.row(r) = data.row(samples[r]);
	}

  //// DBG
  //if (samples.size() < 10) {
  //  std::cout << std::endl << "matrix: " << l << std::endl;
  //  std::cout << "mean: " << NormalDistribution(l).mean() << std::endl;
  //  std::cout << "cov: " << NormalDistribution(l).covariance() << std::endl;
  //}
  //// DBG

	return NormalDistribution(l);
}

bool IsFiniteNumber(double x) {
  return (x <= DBL_MAX && x >= -DBL_MAX); 
} 

inline double RandomForest::DistributionImpurity(const NormalDistribution& nd) {
  double det = nd.covariance().determinant();
  // in case det < 0 something is screwed up - most likely we don't have enough
  // samples to calculate covariance properly, or there's a linear dependence
  // between them - return large impurity to indicate an invalid split
  if (det < 0.0) {
    std::cerr << "Invalid det(cov) : " << det << std::endl;
    return DBL_MAX;
  }
  // if the determinant is 0, this is a perfect case - all samples are the same!
  // that means we have to return perfect impurity score, but what does that mean?
  // we set det to DBL_MIN because we can't compute log of 0
  if (det == 0.0) { det = DBL_MIN; }
  // compute log of the determinant
  double det_log = log(det);
  // sanity check - should never happen!
  if (!IsFiniteNumber(det_log)) {
    std::cerr << "Invalid log(det(cov)) : " << det_log << std::endl;
    return DBL_MAX;
  } 
  return det_log;
};

// calculates impurity of the specified data or distribution
double RandomForest::impurity(
    const Matrix& labels,
    const vector<int>& samples) {
  NormalDistribution nd(getSampleDistr(labels, samples));
	return DistributionImpurity(nd);
};

// construct a matrix as a subset of another
Matrix RandomForest::SubsetRowwise(
    const Matrix& superset,
    const vector<int>& row_indices) {
  int rows = static_cast<int>(row_indices.size());
  Matrix subset(rows, superset.cols());
	for (int i = 0; i < rows; ++i) {
    subset.row(i) = superset.row(row_indices[i]);
	}
  return subset;
}