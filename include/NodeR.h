#ifndef NODER_H
#define NODER_H

#include <vector>

#include "NormalDistribution.h"
#include "RandomForestCommon.h"

// DBG
//#include "DataProvider/PCAWrapper.h"
//#include "DataProvider/DataProvider.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv2/highgui/highgui.hpp>

#define PERFECT_IMPURITY_SCORE (-50.0)
#define BAD_IMPURITY_SCORE (50.0)


namespace RandomForest {

Matrix SubsetRowwise(
    const Matrix& superset,
    const std::vector<int>& row_indices);
// gets mean and covariance of the labels included in samples
NormalDistribution getSampleDistr(
    const Matrix& labels,
    const std::vector<int>& samples);

double DistributionImpurity(const NormalDistribution& nd);

// calculates impurity of the specified data or distribution
double impurity(const Matrix& labels, const std::vector<int>& samples);

// Node in a multivariate regression tree
class NodeR {
public:
	/********** Constructors **********/	
	// initialize NodeR with specified samples
	NodeR(
      int id,
      const Matrix& data,
      const Matrix& labels,
      const std::vector<int>& samples);
	// create a node knowing all the parameters (e.g. read from a file)
	NodeR(
      int id,
      bool isLeaf,
      int split_dim,
      double split_thresh,
      double impurity);
	// create a leaf node knowing all the parameters (e.g. read from a file)
	NodeR(
      int id,
      bool isLeaf,
      int split_dim,
      double split_thresh,
      double impurity,
      const NormalDistribution& nd,
      const std::vector<int>& samples);
	~NodeR();

	/********** Member functions **********/
	// calculates minimum and maximum values for data in this node
	void CalcDimLimits(const Matrix& data);

	// (only for leaf nodes) calculates mean feature vector
	NormalDistribution CalcFeatureDist(const Matrix& data);
  // returns the node that this feature vector ends up at
  NodeR* GetTerminalNode(const RowVector& features);
  // returns the distribution of labels at the leaf node that this 
  // feature vector ends up at
	NormalDistribution EvaluatePoint(const RowVector& features);
  // gets distance in the feature space from this feature vector to the mean of the leaf node
  float GetFeatureDistanceMahalanobis(const RowVector& features, const Matrix& data);

	void WriteNode(std::ofstream& file) const;
	
	int id() const { return id_; };
	double impurity() const { return impurity_; };
	int n_samples() const { return n_samples_; };

	// returns min/max at a given dimension
	double dim_min(int dim) const { return dim_mins_[dim]; };
	double dim_max(int dim) const {	return dim_maxes_[dim]; };

	// returns index of the sample at given position
	int sample(int pos) const { return samples_[pos]; };
  const std::vector<int>& samples() const { return samples_; };

	bool is_leaf() const { return is_leaf_; };
	void set_to_leaf() { is_leaf_ = true; };

	NodeR* left() {
		assert(!is_leaf_);
		return left_;
	};
	
	void set_left(NodeR* left) {
		assert(!is_leaf_);
		left_ = left;
	};

	NodeR* right() {
		assert(!is_leaf_);
		return right_;
	};

	void set_right(NodeR* right) {
		assert(!is_leaf_);
		right_ = right;
	};

  void set_parent(NodeR* parent) { parent_ = parent; };

	void set_split_dim(int split_dim) { split_dim_ = split_dim; };
	void set_split_thresh(double split_thresh) { split_thresh_ = split_thresh; };

  //void WriteNodeLabels(const Matrix& data, const Matrix& labels, const std::string& folder, const std::string& prefix, const PCAWrapper& pcaw, int patch_size) {
  //  if (!is_leaf_) {
  //    left_->WriteNodeLabels(data, labels, folder, prefix, pcaw, patch_size);
  //    right_->WriteNodeLabels(data, labels, folder, prefix, pcaw, patch_size);
  //  } else {
  //    std::stringstream nss;
  //    nss << id_;
  //    std::string node_dir = folder + nss.str() + "\\";
  //    // make sure the directory exists
  //    system(std::string("mkdir " + node_dir).c_str());
  //    // get all labels at this leaf node
  //    Matrix leaf_labels = SubsetRowwise(labels, samples_);
  //    Matrix leaf_features = SubsetRowwise(data, samples_);
  //    
  //    double impurity = RandomForest::DistributionImpurity(NormalDistribution(leaf_labels));
  //    std::string impurity_filename(node_dir + "\\");
  //    std::stringstream ss;
  //    ss << impurity;
  //    impurity_filename += ss.str();
  //    std::ofstream oif;
  //    oif.open(impurity_filename, std::ios::out);
  //    oif << " ";
  //    oif.close();
  //    
  //    DataProvider::serializeMatrixXfAscii(leaf_labels, node_dir + "\\labels.csv");
  //    DataProvider::serializeMatrixXfAscii(leaf_features, node_dir + "\\features.csv");
  //    // back-project from PCA space
  //    leaf_labels = pcaw.BackProject(leaf_labels);
  //    // for each label
  //    for (int r = 0; r < leaf_labels.rows(); ++r) {
  //      // reshape into a square
  //      Matrix im = leaf_labels.row(r);
  //      im.resize(patch_size, patch_size);
  //      // save as an image
  //      cv::Mat im_cv;
  //      cv::eigen2cv(im, im_cv);
  //      std::stringstream rss;
  //      rss << r;
  //      cv::imwrite(node_dir + "\\" + rss.str() + ".png", im_cv*255);
  //    }
  //  }
  //};

  // DBG
  std::vector<float> info_gains_;

private:
	void InitNode(const Matrix& data, const Matrix& labels);

	static std::string matrixString(const Matrix& m);
	static std::string intVectorString(const std::vector<int> vec);

	int id_;	// id of the node (identifies position within 
            // the tree regardless of tree structure)
	int n_samples_;	// number of samples reaching this node
	std::vector<int> samples_;	// indices of samples reaching this node
	
	double impurity_;			// impurity calculated over all samples at this node

	// TODO: some of these are only meaningful at non-leaf nodes, add inheritance?
	int split_dim_;				// dimension of split this node represents
	double split_thresh_;			// threshold of split this node represents

	std::vector<double> dim_mins_;			// minimum and maximum values for each feature
	std::vector<double> dim_maxes_;			// in this node's set of samples
	
	NormalDistribution nd_;		// distribution of labels and this node

	bool is_leaf_;				// true only for terminal nodes

	NodeR* parent_;				// parent node
	NodeR* left_;				// left child node
	NodeR* right_;
	
	// not allowed
	NodeR();
	NodeR(const NodeR&);
};

}  // namespace RandomForest
#endif