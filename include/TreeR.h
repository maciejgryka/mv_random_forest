#ifndef TREER_H
#define TREER_H

#include <vector>

#include "NormalDistribution.h"
#include "RandomForestCommon.h"

namespace RandomForest {

class NodeR;

// Multivariate regression tree
class TreeR {
public:
  // create a tree to be trained
	TreeR(
      int id,
      int max_depth,
      int n_dim_in,
      int n_dim_out,
      int min_samples_count);
  // create a tree knowing all the parameters (read in)
	TreeR(
      int id,
      int max_depth,
      int n_nodes,
      int n_leaves_,
      int n_dim_in,
      int n_dim_out);
	~TreeR();

	//adds a node to the tree
	void AddNodeToTree(int id, bool is_leaf, int split_dim, double split_thresh, double impurity);
	// same as above, but for leaf nodes (also include label distribution)
	void AddNodeToTree(
      int id,
      bool is_leaf,
      int split_dim,
      double split_thresh,
      double impurity,
      const NormalDistribution& nd,
      const std::vector<int>& samples);
  // return pointer to the leaf node where given feature vector lands
  RandomForest::NodeR* GetTerminalNode(const RowVector& features);
  // takes an input feature vector and returns the prediction
  // (mean labels of nodes' suggestions)
  NormalDistribution Test(const RowVector& x);
  // train the tree based on the given data
  void Train(
      const Matrix& data,
      const Matrix& labels,
      const std::vector<int>& bag_samples,
      int n_dim_trials,
      int n_thresh_trials);
  // write the tree to the given output stream
	void WriteTree(std::ofstream& file) const;
  // getters
  int id() const { return id_; };
  int n_leaves() const { return n_leaves_; };
  int n_nodes() const { return n_nodes_; }; 
	int n_dim_out() const { return n_dim_out_; };
  NodeR* root() { return root_; };

private:
  enum ChildType {
    CHILD_LEFT,
    CHILD_RIGHT
  };

  // see above (public) functions with the same name
	bool AddNodeToTree(NodeR* node, NodeR* nodeToAdd);
	// create (left or right) child node connected to parent and update metadata
	void CreateChild(
      NodeR* parent,
      ChildType child_type,
      const Matrix& data,
      const Matrix& labels,
      const std::vector<int>& samples);
  // choose the best split out of n_dim_trials*n_thresh_trials random splits for
  // the current node and recursively optimize all children
	void OptimizeNode(
      NodeR* node,
      int n_dim_trials,
      int n_thresh_trials,
      const Matrix& data,
      const Matrix& labels);
  // this function should be called instead of NodeR::setToLeaf() directly
	// because it also increments leaf count in the tree
	void SetNodeToLeaf(NodeR* node);


  int id_;                // tree ID
	int depth_;			        // current tree depth
	int max_depth_;		      // maximum tree depth
	int n_dim_in_;			    // input dimensionality
	int n_dim_out_;		      // output dimensionality
  int min_sample_count_;  // minimum number of samples at node

	NodeR* root_;
	int n_nodes_;
	int n_leaves_;

  // not allowed
	TreeR();
	TreeR(const TreeR&);
};

std::vector<int> randomIntVector(int size, int min, int max);

} // namespace RandomForest

#endif