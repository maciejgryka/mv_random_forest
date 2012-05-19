#include "RandomForestCommon.h"
#include "Ensemble.h"

using namespace RandomForest;

static float kPi = 3.14159;

// random float between 0.0 and 1.0
float rand1() {
  return static_cast<float>(rand())/RAND_MAX;
}

int main(int argc, char* argv[]) {
  // create training data
  float max_x = 2*kPi;
  int n_samples = 1000;
  int n_dim_in = 1;
  int n_dim_out = 2;
  Matrix features(n_samples, n_dim_in);
	Matrix labels(n_samples, n_dim_out);
  for (int s = 0; s < n_samples; ++s) {
    float x = max_x * rand1();
    features(s, 0) = x;
    labels(s,0) = sin(x);
    labels(s,1) = cos(x);
  }
	
  // define forest parameters
	int n_trees(10);
	int max_depth(10);
	int n_dim_trials(20);
	int n_thresh_trials(100);
	float bag_prob(0.66);
	int min_sample_count(4);
  // create and train the forest
  Ensemble forest(n_trees, max_depth, n_dim_in, n_dim_out);
  forest.Train(
      features,
      labels,
      n_dim_trials,
      n_thresh_trials,
      bag_prob,
      min_sample_count);
  
  // create some test data that's different to training, but in the same range
  int n_samples_test = 100;
	Matrix features_test(n_samples_test, n_dim_in);
	Matrix labels_test_gt(n_samples_test, n_dim_out);
  for (int s = 0; s < n_samples_test; ++s) {
    float x = max_x * rand1();
    features_test(s,0) = x;
    labels_test_gt(s,0) = sin(x);
    labels_test_gt(s,1) = cos(x);
  }
	
  // test on the data and save results
  Matrix labels_test(features_test.rows(), n_dim_out);
  for (int s = 0; s < n_samples_test; ++s) {
    labels_test.row(s) = forest.Test(features_test.row(s));
  }
  SerializeMatrix(features_test, "features_test.csv");
  SerializeMatrix(labels_test_gt, "labels_test_gt.csv");
  SerializeMatrix(labels_test, "labels_test.csv");
} 