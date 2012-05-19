#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include <map>

#include "RandomForestCommon.h"


namespace RandomForest {
class TreeR;
class NodeR;

class Ensemble {	
public:
	Ensemble(int n_trees, int max_depth, int n_dim_in, int n_dim_out);
	Ensemble();
	~Ensemble();

  //std::vector<float> GetUnaryCosts(
  //    const RowVector& features,
  //    const Matrix& data);
  std::vector<float> GetUnaryCostsFromMeanLabel(const Matrix& labels);

  void SetParams(int nTrees, int maxDepth, int nDimIn, int noDimOut);
	
  RowVector Test(const RowVector& x);
	Matrix TestGetAll(const RowVector& x);
  std::vector<NodeR*> GetAllTerminalNodes(const RowVector& x);
	void Train(
      const Matrix& data,
      const Matrix& labels,
      int numDimTrials,
      int numThreshTrials,
      float bagProb,
      int minNoOFExsAtNode);

	void LoadEnsemble(std::string fileName);
	void WriteEnsemble(std::string fileName);

  //// DBG
  //void WriteTreeNodes(int scale, const Matrix& labels, const PCAWrapper& pcaw) {
  //  // DBG
  //  int patch_size = DataProvider::getPatchSize(scale);
  //  // iterate over trees
  //  typedef vector<TreeR*>::iterator TVIter;
  //  for (TVIter it = trees_.begin(); it != trees_.end(); ++it) {
  //    stringstream tss;
  //    tss << (*it)->id();
  //    string tree_dir = "C:\\Projects\\penumbraRemoval\\2012-04-23\\output\\size100\\leaves\\" + tss.str() + "\\";
  //    // write out leaf nodes
  //    (*it)->root()->WriteNodeLabels(labels, tree_dir, string("label_"), pcaw, patch_size);
  //  }
  //};

	int n_dim_in() const { return n_dim_in_; };
	int n_dim_out() const { return n_dim_out_; };
  int n_trees() const { return static_cast<int>(trees_.size()); };
  std::vector<TreeR*> trees() { return trees_; };
private:
  // return a random float in the range 0..1
  float RandFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  };

	int n_trees_;		// number of trees in the forest
	int max_depth_;	// max tree depth
	int n_dim_in_;		// input dimensionality
	int n_dim_out_;	  // output dimensionality
	std::vector<TreeR*> trees_;
};

} // namespace RandomForest

#endif