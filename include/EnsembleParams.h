#ifndef ENSEMBLE_PARAMS_H
#define ENSEMBLE_PARAMS_H

class EnsembleParams {
public:
	EnsembleParams() {};
	EnsembleParams(
      int n_dim_in,
			int n_dim_out, 
			int n_trees, 
			int tree_depth, 
			int n_dim_trials, 
			int n_thresh_trials, 
			float bag_prob, 
			int min_sample_count):

		  n_dim_in_(n_dim_in),
		  n_dim_out_(n_dim_out),
		  n_trees_(n_trees),
		  tree_depth_(tree_depth),
		  n_dim_trials_(n_dim_trials),
		  n_thresh_trials_(n_thresh_trials),
		  bag_prob_(bag_prob),
		  min_sample_count_(min_sample_count) {
		assert(n_dim_in_ > 0 && n_dim_out_ > 0);
		//assert(bag_prob_ > 0.0f && bag_prob_ < 1.0f);
	};

	~EnsembleParams() {};

	int n_dim_in()			       const { return n_dim_in_; };
	int n_dim_out()		         const { return n_dim_out_; };
	int n_trees()			         const { return n_trees_; };
	int tree_depth()		       const { return tree_depth_; };
	int n_dim_trials()		     const { return n_dim_trials_; };
	int n_thresh_trials()	     const { return n_thresh_trials_; };
	float bag_prob()		       const { return bag_prob_; };
	int min_sample_count() const { return min_sample_count_; };

private:
	int n_dim_in_;
	int n_dim_out_;
	int n_trees_;
	int tree_depth_;
	int n_dim_trials_;
	int n_thresh_trials_;
	float bag_prob_;
	int min_sample_count_;
};

#endif