#include "omp.h"
#include "time.h"
#include <vector>
#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include "Ensemble.h"	
#include "NodeR.h"
#include "TreeR.h"
#include "NormalDistribution.h"

using namespace std;
using namespace RandomForest;

Ensemble::Ensemble(int n_trees, int max_depth, int n_dim_in, int n_dim_out):
	  n_trees_(n_trees),
	  max_depth_(max_depth),
	  n_dim_in_(n_dim_in),
	  n_dim_out_(n_dim_out),
    trees_(n_trees, NULL) {}

Ensemble::Ensemble():
    n_trees_(-1),
    n_dim_in_(-1),
    n_dim_out_(-1),
    max_depth_(-1) {}

Ensemble::~Ensemble() {
  for (vector<TreeR*>::iterator it = trees_.begin(); it != trees_.end(); ++it) {
		delete *it;
	}
}

void Ensemble::SetParams(int nTrees, int maxDepth, int nDimIn, int nDimOut) {
	n_trees_ = nTrees;
	n_dim_in_ = nDimIn;
	n_dim_out_ = nDimOut;
	max_depth_ = maxDepth;
  trees_ = vector<TreeR*>(nTrees, NULL);
}

void Ensemble::Train(
    const Matrix& data,
    const Matrix& labels,
    int numDimTrials,
    int numThreshTrials,
    float bagProb,
    int minNoOFExsAtNode) {
	srand(unsigned(time(NULL)));
	//// DBG
 // srand(12345);

  int data_cols = static_cast<int>(data.cols());
  int data_rows = static_cast<int>(data.rows());
  int labels_cols = static_cast<int>(labels.cols());

	vector<int> seeds;
  seeds.reserve(n_trees_);
	for (int t = 0; t < n_trees_; ++t) {
		seeds.push_back(rand());
	}

	n_dim_in_ = data_cols;
	n_dim_out_ = labels_cols;
  trees_ = vector<TreeR*>(n_trees_, NULL);

	#pragma omp parallel for
	for (int eIt = 0; eIt< n_trees_; ++eIt) {
    cout << "\r training tree " << eIt << " of " << n_trees_ << " ";
    //srand(seeds[eIt]);

		vector<int> bag_samples;
    bag_samples.reserve(data_rows);  // worst-case size
		for (int bIt = 0; bIt < data_rows; ++bIt) {
      if (RandFloat() < bagProb) {
				bag_samples.push_back(bIt);
			}
		}
		// initialise new tree and train
		trees_[eIt] = new TreeR(eIt, max_depth_, data_cols, labels_cols, minNoOFExsAtNode);
		trees_[eIt]->Train(data, labels, bag_samples, numDimTrials, numThreshTrials);
	}
}

RowVector Ensemble::Test(const RowVector& x) {
	NormalDistribution nd;
	RowVector mean(n_dim_out());
	mean.setZero();
	for (unsigned t = 0; t < trees_.size(); t++) {
    nd.Update(trees_[t]->Test(x).mean());
	}
  return nd.mean();
}

std::vector<NodeR*> Ensemble::GetAllTerminalNodes(const RowVector& x) {
  std::vector<NodeR*> terminal_nodes(n_trees_);
  for (int t = 0; t < n_trees_; ++t) {
    terminal_nodes[t] = trees_[t]->GetTerminalNode(x);
  }
  return terminal_nodes;
}


// return a matrix where the first row is the mean output from all trees
// and all subsequent rows are outputs from individual trees
Matrix Ensemble::TestGetAll(const RowVector& x) {
	Matrix outs(int(trees_.size()) + 1, n_dim_out());
  outs.row(0).setZero();
	for (int t = 1; t < outs.rows(); ++t) {
		outs.row(t) = trees_[t-1]->Test(x).mean();
    outs.row(0) += outs.row(t);
	}
  outs.row(0) /= float(trees_.size());
	return outs;
}

//vector<float> Ensemble::GetUnaryCosts(
//    const RowVector& features,
//    const Matrix& data) {
//  vector<float> costs(trees_.size());
//  NodeR* leaf;
//  for (int t = 0; t < int(trees_.size()); ++t) {
//    leaf = trees_[t]->GetTerminalNode(features);
//    costs[t] = leaf->GetFeatureDistanceMahalanobis(features, data);
//  }
//  return costs;
//}

vector<float> Ensemble::GetUnaryCostsFromMeanLabel(const Matrix& labels) {
  vector<float> costs(labels.rows());
  for (int l = 1; l < labels.rows(); ++l) {
    costs[l] = (labels.row(0) - labels.row(l)).norm();
  }
  return costs;
}

void Ensemble::WriteEnsemble(string fileName) {
	ofstream opFile;
	opFile.open(fileName, ios::out);

	assert(opFile.good());

	opFile << n_trees_ << endl;
	opFile << n_dim_in_ << endl;
	opFile << n_dim_out_ << endl << endl;

	for (unsigned tIt = 0; tIt < trees_.size(); tIt ++) {
		opFile << tIt << endl;
		trees_[tIt]->WriteTree(opFile);
		opFile << endl;
	}
	opFile.close();
}

void Ensemble::LoadEnsemble(string fileName) {

	ifstream infile;	
	infile.open(fileName, ios::in);	
	string line;

	// delete current trees if there are any
	trees_.clear();
	
	istringstream istr;

	// read in the ensemble params
	int ensembleParams = 3;
	vector<int> readVals(ensembleParams);
	
	for (int i = 0; i < ensembleParams; i++) {
		getline(infile,line);
		istringstream(line) >> readVals[i];
	}

	// global vars for this class
	n_trees_  = readVals[0];
	n_dim_in_  = readVals[1];
	n_dim_out_ = readVals[2];

	getline(infile,line);	// blank line
	
	// read in the trees
	for(int tIt = 0; tIt < n_trees_; tIt++) {		
		int tree_id;
		getline(infile,line);	// tree number
		istringstream(line) >> tree_id;

		int maxDepth;
		getline(infile,line);	// tree depth
		istr = istringstream(line);
		istr >> maxDepth;

		int nNodes;
		getline(infile,line);	// # of nodes
		istr = istringstream(line);
		istr >> nNodes;

		int nLeaves;
		getline(infile,line);	// # of leaves
		istr = istringstream(line);
		istr >> nLeaves;

		// create a new tree
		TreeR* tr = new TreeR(tree_id, maxDepth, nNodes, nLeaves, n_dim_in_, n_dim_out_);

		// read in node vals
		for (int j=0; j < nNodes; j++) {
			int id;
			getline(infile,line);	// node id
			istr = istringstream(line);
			istr >> id;

			bool isLeaf;
			getline(infile,line);
			istr = istringstream(line);
			istr >> isLeaf;

      int nSamples;
			getline(infile,line);
      istr = istringstream(line);
      istr >> nSamples;

			int splitDim;
			getline(infile,line);	// split dimension
			istr = istringstream(line);
			istr >> splitDim;

			double splitThresh;
			getline(infile,line);	// split threshold
			istr = istringstream(line);
			istr >> splitThresh;

      double impurity;
      getline(infile,line);
			istr = istringstream(line);
      istr >> impurity;

			// if this is leaf, also read label distribution and samples that landed here
			if (isLeaf) {
        getline(infile,line);	// node mean
			  RowVector mean(ReadFloatVector(line, n_dim_out_));
			
			  getline(infile,line);	// node covariance
			  Matrix cov(ReadFloatMatrix(line, n_dim_out_, n_dim_out_));

				getline(infile, line);
        vector<int> samples(RandomForest::StringToStdVector<int>(line, nSamples));
			  tr->AddNodeToTree(id, isLeaf, splitDim, splitThresh, impurity, NormalDistribution(mean, cov), samples);
			} else {
  			getline(infile,line);	// information gains for all evaluated splits
        vector<float> info_gains(RandomForest::StringToStdVector<float>(line));
				tr->AddNodeToTree(id, isLeaf, splitDim, splitThresh, impurity);
			}
		}
		trees_.push_back(tr);
		getline(infile,line);	// empty line
	}
}
