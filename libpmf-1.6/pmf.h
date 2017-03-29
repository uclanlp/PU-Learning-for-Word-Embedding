#ifndef _PMF_H_
#define _PMF_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <cmath>
#include <assert.h>
#include <vector>
#include <limits>
#include <omp.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define puts(str) mexPrintf("%s\n",(str))
#define fflush(str) mexEvalString("drawnow")
#endif

#ifdef WITH_SMAT_H
#include "smat.h"
#else
#include "sparse_matrix.h"
#include "block_matrix.h"
#ifdef _USE_FLOAT_
#define val_type float
#else
#define val_type double
#endif
typedef sparse_matrix<val_type> smat_t;
typedef block_matrix<val_type> blocks_t;
#endif // end of ifdef WITH_SMAT_H


//typedef std::vector<val_type> vec_t;
//typedef std::vector<vec_t> mat_t;
typedef dense_vector<val_type> vec_t;
typedef dense_matrix<val_type> mat_t;

/* solver types*/
enum {CCDR1=0, ALS=1, SGD=2, CCDR1_SPEEDUP=9, PU_CCDR1=10, PU_ALS=11, PU_SGD=12, PU_CCDR1_SPEEDUP=19, PU_SGD_ORIG=22};

enum {PU0=0, PU1=1, PU2=2, PU3=3, PU4=4, PU5=5, PU6=6, PU7=7, PU8=8};


// pmf model
class pmf_model_t {//{{{
	public:
		size_t rows, cols;
		size_t k;
		mat_t W, H;
		val_type global_bias;
		enum major_t {ROWMAJOR, COLMAJOR};
		int major_type;

		pmf_model_t(major_t major_type_=COLMAJOR): major_type(major_type_){}
		pmf_model_t(size_t rows_, size_t cols_, size_t k_, major_t major_type_, bool do_rand_init=true, val_type global_bias_=0.0);
		void rand_init(long seed=0L);

		val_type predict_entry(size_t i, size_t j) const;
		template<typename T, typename T2>
		void predict_entries(size_t nr_entries, const T *row_idx, const T *col_idx, T2 *pred_val, int idx_base=0, int nr_threads=0) const { // {{{
			if (nr_threads == 0)
				nr_threads = omp_get_max_threads();
			omp_set_num_threads(nr_threads);
#pragma omp parallel for schedule(static)
			for(long i = 0; i < nr_entries; i++) {
				pred_val[i] = (T2) predict_entry((size_t)row_idx[i]-idx_base, (size_t)col_idx[i]-idx_base);
			}
		} // }}}

		template<typename T>
		void predict_row(size_t r, size_t nr_entries, T *col_idx, double *pred_val, int idx_base=0) const { // {{{
			for(size_t i = 0; i < nr_entries; i++)  {
				size_t c = (size_t)(col_idx[i]-idx_base);
				pred_val[c] = predict_entry(r, c);
			}
		} // }}}

		template<typename T>
		void predict_col(size_t c, size_t nr_entries, T *row_idx, double *pred_val, int idx_base=0) const { // {{{
			for(size_t i = 0; i < nr_entries; i++) {
				size_t r = (size_t)(row_idx[i]-idx_base);
				pred_val[r] = predict_entry(r, c);
			}
		} // }}}

		void apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm);
		void apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL);
		void save(FILE *fp);
		void save_embedding(FILE *fpw, FILE *fph);
		void load(FILE *fp, major_t major_type_);
	private:
		void mat_rand_init(mat_t &X, size_t m, size_t n, long seed);
};//}}}

class pmf_parameter_t {//{{{
	public:
		int solver_type;
		long k;
		int threads;
		int maxiter, maxinneriter;
		double lambda;
		double rho;
		double eps;						// for the fundec stop-cond in ccdr1
		double eta0, betaup, betadown;  // learning rate parameters used in DSGD
		double alpha;
		int lrate_method, nr_blocks;
		int pu_type; // pu types for sgd
		int warm_start;
		int remove_bias;
		int do_predict, verbose;
		int do_nmf;  // non-negative matrix factorization
		pmf_parameter_t() {
			solver_type = CCDR1;
			k = 10;
			rho = 1e-3;
			maxiter = 5;
			maxinneriter = 5;
			lambda = 0.1;
			threads = 4;
			eps = 1e-3;
			/* options for dsgd
			betaup = 1.05;
			betadown = 0.5;
			lrate_method = BOLDDRIVER;
			*/
			eta0 = 0.1; // initial eta0
			nr_blocks = 0;  // number of blocks used in dsgd
			pu_type = PU2;
			alpha = 0.01; // sampling rate

			warm_start = 0;
			remove_bias = 1;
			do_predict = 0;
			verbose = 0;
			do_nmf = 0;
		}
};//}}}

/* solvers using COLMAJOR*/
void ccdr1(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model);
void ccdr1_pu(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model);
void ccdr1_speedup(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model);

/* solvers using ROWMAJOR*/
void als(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model);
void als_pu(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model);

void sgd(blocks_t &training_set, blocks_t &test_set, pmf_parameter_t &param, pmf_model_t &model);
void sgd_pu(blocks_t &training_set, blocks_t &test_set, pmf_parameter_t &param, pmf_model_t &model);

/*utility functions*/
void load_mat_t(FILE *fp, mat_t &A, bool row_major=false);
void save_wordembedding(const mat_t &A, FILE *fp, bool row_major=false);
void save_mat_t(const mat_t &A, FILE *fp, bool row_major=false);
void pmf_read_data(const char* srcdir, smat_t &training_set, smat_t &test_set, smat_t::format_t fmt = smat_t::TXT);
void pmf_read_data(const char* srcdir, blocks_t &training_set, blocks_t &test_set, smat_t::format_t fmt = smat_t::TXT);


/* random number genrator: simulate the interface of python random module*/
#include <algorithm>
#include <limits>
#if __cpluscplus >= 201103L || (defined(_MSC_VER) && (_MSC_VER >= 1500)) // Visual Studio 2008
#include <random>
template<typename engine_t=std::mt19937>
struct random_number_generator : public engine_t { // {{{
	typedef typename engine_t::result_type result_type;

	random_number_generator(unsigned seed=0): engine_t(seed){ }

	result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
	template<class T> T uniform(T start=0.0, T end=1.0) {
		return std::uniform_real_distribution<T>(start, end)(*this);
	}
	template<class T> T normal(T mean=0.0, T stddev=1.0) {
		return std::normal_distribution<T>(mean, stddev)(*this);
	}
	template<class T> T randint(T start=0, T end=std::numeric_limits<T>::max()) {
		return std::uniform_int_distribution<T>(start, end)(*this);
	}
	template<class RandIter> void shuffle(RandIter first, RandIter last) {
		std::shuffle(first, last, *this);
	}
};
#else
#include <tr1/random>
template<typename engine_t=std::tr1::mt19937>
struct random_number_generator : public engine_t {
	typedef typename engine_t::result_type result_type;

	random_number_generator(unsigned seed=0): engine_t(seed) { }
	result_type operator()() { return engine_t::operator()(); }
	result_type operator()(result_type n) { return randint(result_type(0), result_type(n-1)); }

	result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
	template<class T> T uniform(T start=0.0, T end=1.0) {
		typedef std::tr1::uniform_real<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,end))();
	}
	template<class T> T normal(T mean=0.0, T stddev=1.0) {
		typedef std::tr1::normal_distribution<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(mean, stddev))();
	}
	template<class T> T randint(T start=0, T end=std::numeric_limits<T>::max()) {
		typedef std::tr1::uniform_int<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,end))();
	}
	template<class RandIter> void shuffle(RandIter first, RandIter last) {
		std::random_shuffle(first, last, *this);
	}
}; // }}}
#endif
typedef random_number_generator<> rng_t;

template<typename T>
void gen_permutation_pair(size_t size, std::vector<T> &perm, std::vector<T> &inv_perm, int seed=0) { // {{{
	perm.resize(size);
	for(size_t i = 0; i < size; i++)
		perm[i] = i;

	rng_t rng(seed);
	rng.shuffle(perm.begin(), perm.end());
	//std::srand(seed);
	//std::random_shuffle(perm.begin(), perm.end());

	inv_perm.resize(size);
	for(size_t i = 0; i < size; i++)
		inv_perm[perm[i]] = i;
} // }}}


// Ranking Evaluation Utility Functions

// input: pred_val is a double array of length=len
//        idx is an size_t array of length=len with 0,1,...,len-1 as its elements
// output: the topk elements of idx is sorted according the decreasing order of elements of pred_val.
void sort_idx_by_val(const double *pred_val, size_t len, size_t *idx, size_t topk);

// Initialize pred_val to -inf for ignored indices and 0 for others
// Initialize candidates with 0,...,len-1
template<typename T>
void pmf_prepare_candidates(size_t len, double *pred_val, size_t *candidates, size_t &valid_len, size_t nr_ignore = 0, T *ignore_list=NULL) {  // {{{
	const double Inf = std::numeric_limits<double>::infinity();
	for(size_t i = 0; i < len; i++) {
		pred_val[i] = 0;
		candidates[i] = i;
	}
	if(nr_ignore != 0 && ignore_list != NULL)
		for(size_t i = 0; i < nr_ignore; i++) {
			long ignored_idx = (long) ignore_list[i];
			if(ignored_idx >= 0 && ignored_idx < len)
				pred_val[(long)ignore_list[i]] = -Inf;
		}

	valid_len = len;
	for(size_t i = 0; i < valid_len; i++)
		if(pred_val[candidates[i]] < 0) {
			std::swap(candidates[i], candidates[valid_len-1]);
			valid_len--;
			i--;
		}
} // }}}

// input: idx is an sorted index array of length=len
// output: dcg is the array of length=topk with accumuated dcg information
// return: dcg@topk
double compute_dcg(const double *true_rel, size_t *sorted_idx, size_t len, int topk, double *dcg=NULL);

struct info_t { // {{{
	std::vector<size_t> sorted_idx;
	std::vector<double> true_rel;
	std::vector<double> pred_val;
	std::vector<double> tmpdcg, maxdcg;
	std::vector<double> dcg, ndcg;
	std::vector<size_t> count;
	double map, auc, hlu;
}; // }}}

// for matlab
template<typename T1, typename T2, typename T3, typename T4>
void compute_ndcg_csc(size_t rows, size_t cols, T1 *col_ptr, T2 *row_idx, T3 *true_rel, T4 *pred_val, int topk, double *dcg, double *ndcg) { // {{{
	int nr_threads = omp_get_num_procs();

	omp_set_num_threads(nr_threads);
	std::vector<info_t> info_set(nr_threads);

	for(int th = 0; th < nr_threads; th++) {
		info_set[th].sorted_idx.reserve(cols);
		info_set[th].true_rel.reserve(cols);
		info_set[th].pred_val.reserve(cols);
		info_set[th].tmpdcg.resize(topk);
		info_set[th].maxdcg.resize(topk);
		info_set[th].dcg.resize(topk);
		info_set[th].ndcg.resize(topk);
		info_set[th].count.resize(topk);
	}

//#pragma omp parallel for schedule(dynamic,16)
	for(int i = 0; i < cols; i++) {
		size_t len = (size_t) (col_ptr[i+1]-col_ptr[i]);
		if(len == 0) continue;

		int tid = omp_get_thread_num();
		info_t &info = info_set[tid];
		info.sorted_idx.resize(len);
		info.true_rel.resize(len);
		info.pred_val.resize(len);

		size_t valid_len = 0;
		for(size_t idx = 0; idx < len; idx++) {
			double yi = (double) true_rel[col_ptr[i]+idx];
			double pi = (double) pred_val[col_ptr[i]+idx];
			if(!std::isnan(yi) && !std::isnan(pi)) {
				info.sorted_idx[valid_len] = valid_len;
				info.true_rel[valid_len] = yi;
				info.pred_val[valid_len] = pi;
				valid_len++;
			}
		}
		if(valid_len == 0) continue;

		sort_idx_by_val(info.pred_val.data(), valid_len, info.sorted_idx.data(), topk);
		compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.tmpdcg.data());

		sort_idx_by_val(info.true_rel.data(), valid_len, info.sorted_idx.data(), topk);
		compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.maxdcg.data());

		for(int k = 0; k < topk; k++) {
			double tmpdcg = info.tmpdcg[k];
			double tmpmaxdcg = info.maxdcg[k];
			if(std::isfinite(tmpdcg) && std::isfinite(tmpmaxdcg) && tmpmaxdcg>0) {
				info.dcg[k] += tmpdcg;
				info.ndcg[k] += 100*tmpdcg/tmpmaxdcg;
				info.count[k] ++;
			}
		}
	}

	// aggregate results from multiple threads
	info_t &final_info = info_set[0];
	for(int th = 1; th < nr_threads; th++) {
		info_t &info = info_set[th];
		for(int k = 0; k < topk; k++) {
			final_info.dcg[k] += info.dcg[k];
			final_info.ndcg[k] += info.ndcg[k];
			final_info.count[k] += info.count[k];
		}
	}
	for(int k = 0; k < topk; k++) {
		dcg[k] = final_info.dcg[k] / (double) final_info.count[k];
		ndcg[k] = final_info.ndcg[k] / (double) final_info.count[k];
		//if(final_info.count[k] < cols) { printf("Warning: skip %ld groups with NaN/Inf values\n", cols - final_info.count[k]); }
	}
} // }}}

template<typename T1, typename T2, typename T3, typename T4>
void compute_ndcg_csr(size_t rows, size_t cols, T1 *row_ptr, T2 *col_idx, T3 *true_rel, T4 *pred_val, int topk, double *dcg, double *ndcg) { // {{{
	compute_ndcg_csc(cols, rows, row_ptr, col_idx, true_rel, pred_val, topk, dcg, ndcg);
} // }}}

template<typename T1, typename T2, typename T3, typename T4>
void compute_ndcg_csr_full(size_t rows, size_t cols, T1 *row_ptr, T2 *col_idx, T3 *val_t, T4 *pred_topk, int topk, double *dcg, double *ndcg, int idx_base=0) { // {{{
	int nr_threads = omp_get_num_procs();

	//printf("nr_threads %d\n", nr_threads);
	omp_set_num_threads(nr_threads);
	std::vector<info_t> info_set(nr_threads);

	for(int th = 0; th < nr_threads; th++) {
		info_set[th].sorted_idx.reserve(cols);
		info_set[th].true_rel.resize(cols, 0);
		info_set[th].tmpdcg.resize(topk);
		info_set[th].maxdcg.resize(topk);
		info_set[th].dcg.resize(topk);
		info_set[th].ndcg.resize(topk);
		info_set[th].count.resize(topk);
	}
#pragma omp parallel for
	for(long i = 0; i < rows; i++) {
		info_t &info = info_set[omp_get_thread_num()];
		double *true_rel = info.true_rel.data();
		info.sorted_idx.resize(topk);
		size_t *sorted_idx = info.sorted_idx.data();

		for(size_t idx = row_ptr[i]; idx != row_ptr[i+1]; idx++)
			true_rel[col_idx[idx]] += val_t[idx];

		size_t valid_len = 0;
		for(int t = 0; t < topk; t++) {
			long tmp_idx = (long) (pred_topk[i*topk+t]-idx_base);
			if(tmp_idx >= 0) {
				sorted_idx[t] = tmp_idx;
				valid_len ++;
			}
		}
		compute_dcg(true_rel, sorted_idx, valid_len, topk, info.tmpdcg.data());
		valid_len = row_ptr[i+1] - row_ptr[i];
		if(valid_len) {
			info.sorted_idx.resize(valid_len);
			sorted_idx = info.sorted_idx.data();
			for(size_t idx = row_ptr[i],t=0; idx != row_ptr[i+1]; idx++,t++)
				sorted_idx[t] = (size_t) col_idx[idx];
			sort_idx_by_val(true_rel, valid_len, sorted_idx, topk);
			compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.maxdcg.data());

			for(int k = 0; k < topk; k++) {
				double tmpdcg = info.tmpdcg[k];
				double tmpmaxdcg = info.maxdcg[k];
				if(std::isfinite(tmpdcg) && std::isfinite(tmpmaxdcg) && tmpmaxdcg>0) {
					info.dcg[k] += tmpdcg;
					info.ndcg[k] += 100*tmpdcg/tmpmaxdcg;
					info.count[k] ++;
				}
			}
		}
		for(size_t idx = row_ptr[i]; idx != row_ptr[i+1]; idx++)
			true_rel[col_idx[idx]] -= val_t[idx];
	}

	// aggregate results from multiple threads
	info_t &final_info = info_set[0];
	for(int th = 1; th < nr_threads; th++) {
		info_t &info = info_set[th];
		for(int k = 0; k < topk; k++) {
			final_info.dcg[k] += info.dcg[k];
			final_info.ndcg[k] += info.ndcg[k];
			final_info.count[k] += info.count[k];
		}
	}

	for(int k = 0; k < topk; k++) {
		dcg[k] = final_info.dcg[k] / (double) final_info.count[k];
		ndcg[k] = final_info.ndcg[k] / (double) final_info.count[k];
	}
} // }}}

template<typename T1, typename T2, typename T3, typename T4>
void compute_ndcg_csc_full(size_t rows, size_t cols, T1 *col_ptr, T2 *row_idx, T3 *val, T4 *pred_topk, int topk, double *dcg, double *ndcg, int idx_base=0) { // {{{
	compute_ndcg_csr_full(rows, cols, col_ptr, row_idx, val, pred_topk, topk, dcg, ndcg, idx_base);
} // }}}
#endif // end of _PMF_H
