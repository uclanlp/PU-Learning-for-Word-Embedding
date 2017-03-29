#include "pmf.h"
#define kind dynamic,64

#define kSLOW 8UL

typedef blocks_t::rate_t rate_t;
typedef blocks_t::entry_set_t entry_set_t;
typedef blocks_t::block_t block_t;

#ifdef _MSC_VER // {{{
int rand_r(unsigned int *seed) {
	unsigned int next = *seed;
	int result;

	next *= 1103515245;
	next += 12345;
	result = (unsigned int) (next / 65536) % 2048;

	next *= 1103515245;
	next += 12345;
	result <<= 10;
	result ^= (unsigned int) (next / 65536) % 1024;

	next *= 1103515245;
	next += 12345;
	result <<= 10;
	result ^= (unsigned int) (next / 65536) % 1024;

	*seed = next;

	return result;
}
#endif // }}}

static inline val_type dot(const vec_t& Wi, const vec_t& Hj) { // {{{
	val_type ret = 0.0;
	const size_t k = Wi.size();
	for(size_t t = 0; t < k; t++)
		ret += Wi[t]*Hj[t];
	return ret;
} // }}}

static inline val_type dot(const val_type *Wi, const val_type *Hj, const size_t k) { // {{{
//static inline val_type dot(const val_type * __restrict__ Wi, const val_type * __restrict__ Hj, const size_t k) { 
	val_type ret = 0.0;
	for(size_t t = 0; t < k; t++)
		ret += *(Wi++)*(*Hj++);
	return ret;
} // }}}

static double compute_loss(const blocks_t &blocks, const pmf_parameter_t &param, const pmf_model_t &model) { // {{{
	const mat_t &W = model.W, &H = model.H;
	val_type loss = val_type(0.0);
#pragma omp parallel for schedule(kind) reduction(+:loss)
	for(int bid = 0; bid < blocks.size(); bid++) {
		const block_t &block = blocks[bid];
		val_type loss_inner = 0;
		for(size_t idx = 0; idx < block.size(); idx++) {
			const rate_t &r = block[idx];
			val_type diff = dot(W[r.i],H[r.j])-r.v;
			loss_inner += diff*diff;
		}
		loss += loss_inner;
	}
	return loss;
} // }}}

static double compute_loss_orig(const blocks_t &blocks, const pmf_parameter_t &param, const pmf_model_t &model) { // {{{
	const mat_t &W = model.W, &H = model.H;
	double loss = double(0.0);
#pragma omp parallel for schedule(kind) reduction(+:loss)
	for(int bid = 0; bid < blocks.size(); bid++) {
		const block_t &block = blocks[bid];
		double loss_inner = 0;
		for(size_t idx = 0; idx < block.size(); idx++) {
			const rate_t &r = block[idx];
			double sum = dot(W[r.i],H[r.j]);
			val_type diff = sum-r.v;
			if(r.v > 0)
				loss_inner += diff*diff;
			else 
				loss_inner += param.rho*sum*sum;
		}
		loss += loss_inner;
	}
	return loss;
} // }}}

static double compute_pu_loss(const blocks_t &blocks, const pmf_parameter_t &param, pmf_model_t &model, double *loss_omega, double *loss_zero) { // {{{
	const mat_t &W = model.W, &H = model.H;
	vec_t omega_parts(param.threads), zero_parts(param.threads);
	memset(omega_parts.data(), 0, param.threads*sizeof(val_type));
	memset(zero_parts.data(), 0, param.threads*sizeof(val_type));
#pragma omp parallel for schedule(kind)
	for(int bid = 0; bid < blocks.size(); bid++) {
		val_type omega_part = val_type(0.0), zero_part = val_type(0.0);
		const block_t &block = blocks[bid];
		val_type loss_inner = 0;
		for(size_t idx = 0; idx < block.size(); idx++) {
			const rate_t &r = block[idx];
			val_type sum = dot(W[r.i],H[r.j]);
			zero_part -= sum*sum;
			sum -= r.v;
			omega_part += sum*sum;
		}
		omega_parts[omp_get_thread_num()] += omega_part;
		zero_parts[omp_get_thread_num()] += zero_part;
	}
	val_type omega_part = val_type(0.0), zero_part = val_type(0.0);
	for(size_t tid = 0; tid < param.threads; tid++) {
		omega_part += omega_parts[tid];
		zero_part += zero_parts[tid];
	}

	vec_t HTH(model.k*model.k), WTW(model.k*model.k);
	doHTH(model.W.data(), WTW.data(), model.rows, model.k);
	doHTH(model.H.data(), HTH.data(), model.cols, model.k);
	zero_part += do_dot_product(WTW.data(), HTH.data(), model.k*model.k);
	zero_part *= param.rho;

	if(loss_omega) *loss_omega = (double)omega_part;
	if(loss_zero) *loss_zero = (double)zero_part;

	return (double)omega_part+zero_part;
} // }}}

static double compute_reg(const blocks_t &blocks, const pmf_parameter_t &param, const pmf_model_t &model) { // {{{
	double norm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:norm)
	for(long r = 0; r < model.rows; r++) {
		const vec_t &Wi = model.W[r];
		for(size_t t = 0; t < model.k; t++)
			norm += Wi[t]*Wi[t]*blocks.nnz_of_row(r);
	}
#pragma omp parallel for schedule(static) reduction(+:norm)
	for(long c = 0; c < model.cols; c++) {
		const vec_t &Hj = model.H[c];
		for(size_t t = 0; t < model.k; t++)
			norm += Hj[t]*Hj[t]*blocks.nnz_of_col(c);
	}
	return norm;
} // }}}


// See http://shelfflag.com/rsqrt.pdf
static float rsqrt(float number) { // {{{
	unsigned i;
	float x2, y;
	x2 = number * 0.5F;
	y = number;
	i = *(unsigned *) &y;
	i = 0x5f375a86 - (i >> 1);
	y = *(float *) &i;
	y = y * (1.5F - (x2 * y * y));
	return y;
} // }}}
static double rsqrt(double number) { // {{{
	size_t i;
	double x2, y;
	x2 = number * 0.5;
	y = number;
	i = *(size_t *) &y;
	i = 0x5fe6eb50c7b537a9 - (i >> 1);
	y = *(double *) &i;
	y = y * (1.5 - (x2 * y * y));
	return y;
} // }}}
struct sg_info_t { // {{{
	struct grad_info_t {
		val_type slow, fast;
		grad_info_t(val_type slow=1.0, val_type fast=1.0): slow(slow), fast(fast) {}
	};
	std::vector<grad_info_t> grad_w, grad_h;
	std::vector<size_t> counts;
	bool slow_only;
	size_t k, k_slow;
	val_type invk_fast, invk_slow;
	size_t tt;

	std::vector<vec_t> HTH, WTW, WHTH, HWTW; 

	sg_info_t(): slow_only(true) {}
	sg_info_t(const blocks_t &blocks, const pmf_model_t &model, int threads=1): slow_only(true) {
		grad_w.resize(blocks.rows); grad_h.resize(blocks.cols);
		counts.resize(blocks.size(), 0);
		k = model.k;
		k_slow = std::min(k, k/kSLOW);
		invk_fast = 1.0/(k-k_slow);
		invk_slow = 1.0/k_slow;
		tt = 0;
		HTH.resize(threads, vec_t(k*k));
		WTW.resize(threads, vec_t(k*k));
		WHTH.resize(threads, vec_t(model.rows*k));
		HWTW.resize(threads, vec_t(model.cols*k));
	}

}; // }}}

static inline void sgd_core_orig(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t &info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;
	size_t k = model.k;

	size_t blocknnz = block.size();
	while(maxiter--){
		for(size_t idx = 0; idx < blocknnz; idx++) {
			info.counts[bid]++;
			const rate_t &r = block[idx];
			vec_t &Wi = W[r.i], &Hj = H[r.j];
			val_type err = dot(Wi,Hj)-r.v;

			for(size_t t = 0; t < k; t++) {
				val_type tmp = Wi[t];
				Wi[t] -= lrate*(err*Hj[t]+lambda*Wi[t]);
				Hj[t] -= lrate*(err*tmp+lambda*Hj[t]);
				if(param.do_nmf) {
					if(Wi[t] < 0.0) Wi[t] = 0;
					if(Hj[t] < 0.0) Hj[t] = 0;
				}
			}
		}
	}
} // }}}

static inline void sgd_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;
	size_t k = model.k;
	size_t k_slow = std::min(k, k/kSLOW);
	val_type invk_fast = 1.0/(k-k_slow), invk_slow = 1.0/k_slow;

	size_t blocknnz = block.size();
	while(maxiter--){
		for(size_t idx = 0; idx < blocknnz; idx++) {
			info.counts[bid]++;
			const rate_t &r = block[idx];
			//vec_t &Wi = W[r.i], &Hj = H[r.j];
			//val_type err = dot(Wi,Hj)-r.v;
			val_type *Wi = W[r.i].data(), *Hj = H[r.j].data();
			val_type err = dot(Wi,Hj,k)-r.v;

			val_type lrate_w_slow=0, lrate_h_slow=0, grad_w_slow=0, grad_h_slow=0;
			val_type lrate_w_fast=0, lrate_h_fast=0, grad_w_fast=0, grad_h_fast=0;
			sg_info_t::grad_info_t &gw = info.grad_w[r.i], &gh = info.grad_h[r.j];

			if(k_slow > 0) {
				lrate_w_slow = lrate*rsqrt(gw.slow); 
				lrate_h_slow = lrate*rsqrt(gh.slow);
			}
			if(k_slow < k) {
				lrate_w_fast = lrate*rsqrt(gw.fast); 
				lrate_h_fast = lrate*rsqrt(gh.fast);
			}

			{ //  twin learners   {{{
				for(size_t t = 0; t < k_slow; t++) {
					val_type gw = err*Hj[t]+lambda*Wi[t];
					val_type gh = err*Wi[t]+lambda*Hj[t];
					grad_w_slow += gw*gw;
					grad_h_slow += gh*gh;
					Wi[t] -= lrate_w_slow*gw;
					Hj[t] -= lrate_h_slow*gh;
					if(param.do_nmf) {
						if(Wi[t] < 0.0) Wi[t] = 0;
						if(Hj[t] < 0.0) Hj[t] = 0;
					}
				}
				if(!info.slow_only) // ???
				for(size_t t = k_slow; t < k; t++) {
					val_type gw = err*Hj[t]+lambda*Wi[t];
					val_type gh = err*Wi[t]+lambda*Hj[t];
					grad_w_fast += gw*gw;
					grad_h_fast += gh*gh;
					Wi[t] -= lrate_w_fast*gw;
					Hj[t] -= lrate_h_fast*gh;
					if(param.do_nmf) {
						if(Wi[t] < 0.0) Wi[t] = 0;
						if(Hj[t] < 0.0) Hj[t] = 0;
					}
				}
				gw.slow += grad_w_slow*invk_slow;
				gh.slow += grad_h_slow*invk_slow;
				if(!info.slow_only) {
					gw.fast += grad_w_fast*invk_fast;
					gh.fast += grad_h_fast*invk_fast;
				}
			} // }}}
		}
	}
} // }}}

static inline void sgd_core_adgrad_weighted(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;
	val_type rho = param.rho;
	size_t rows = training_set.rows;
	size_t cols = training_set.cols;
	size_t k = model.k;
	size_t k_slow = std::min(k, k/kSLOW);
	val_type invk_fast = 1.0/(k-k_slow), invk_slow = 1.0/k_slow;

	size_t blocknnz = block.size();
	while(maxiter--){
		for(size_t idx = 0; idx < blocknnz; idx++) {
			info.counts[bid]++;
			const rate_t &r = block[idx];
			//vec_t &Wi = W[r.i], &Hj = H[r.j];
			//val_type err = dot(Wi,Hj)-r.v;
			val_type *Wi = W[r.i].data(), *Hj = H[r.j].data();
			val_type lambda_w = lambda*training_set.nnz_of_row(r.i)/cols;
			val_type lambda_h = lambda*training_set.nnz_of_col(r.j)/rows;
			val_type err = dot(Wi,Hj,k)-r.v;
			if(r.v < 0.5) 
				err *= rho;

			val_type lrate_w_slow=0, lrate_h_slow=0, grad_w_slow=0, grad_h_slow=0;
			val_type lrate_w_fast=0, lrate_h_fast=0, grad_w_fast=0, grad_h_fast=0;
			sg_info_t::grad_info_t &gw = info.grad_w[r.i], &gh = info.grad_h[r.j];

			if(k_slow > 0) {
				lrate_w_slow = lrate*rsqrt(gw.slow); 
				lrate_h_slow = lrate*rsqrt(gh.slow);
			}
			if(k_slow < k) {
				lrate_w_fast = lrate*rsqrt(gw.fast); 
				lrate_h_fast = lrate*rsqrt(gh.fast);
			}

			{ //  twin learners   {{{
				for(size_t t = 0; t < k_slow; t++) {
					val_type gw = err*Hj[t]+lambda_w*Wi[t];
					val_type gh = err*Wi[t]+lambda_h*Hj[t];
					grad_w_slow += gw*gw;
					grad_h_slow += gh*gh;
					Wi[t] -= lrate_w_slow*gw;
					Hj[t] -= lrate_h_slow*gh;
					if(param.do_nmf) {
						if(Wi[t] < 0.0) Wi[t] = 0;
						if(Hj[t] < 0.0) Hj[t] = 0;
					}
				}
				if(!info.slow_only) // ???
				for(size_t t = k_slow; t < k; t++) {
					val_type gw = err*Hj[t]+lambda_w*Wi[t];
					val_type gh = err*Wi[t]+lambda_h*Hj[t];
					grad_w_fast += gw*gw;
					grad_h_fast += gh*gh;
					Wi[t] -= lrate_w_fast*gw;
					Hj[t] -= lrate_h_fast*gh;
					if(param.do_nmf) {
						if(Wi[t] < 0.0) Wi[t] = 0;
						if(Hj[t] < 0.0) Hj[t] = 0;
					}
				}
				gw.slow += grad_w_slow*invk_slow;
				gh.slow += grad_h_slow*invk_slow;
				if(!info.slow_only) {
					gw.fast += grad_w_fast*invk_fast;
					gh.fast += grad_h_fast*invk_fast;
				}
			} // }}}
		}
	}
} // }}}

void sgd(blocks_t &training_set, blocks_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	if(model.major_type != pmf_model_t::ROWMAJOR) {
		fprintf(stderr, "SGD requires pmf_model_t::ROWMAJOR model\n");
		return;
	}

	int maxiter = param.maxiter;
	int num_threads_old = omp_get_num_threads();
	size_t k = param.k;
	double lambda = param.lambda;
	double lrate = param.eta0;
	int innerB = param.nr_blocks;

	if(innerB <= 0) innerB = 4*param.threads;

	bool row_major = (training_set.rows > training_set.cols)? true : false;
	training_set.set_blocks(innerB, row_major);
	test_set.set_blocks(innerB, row_major);
	mat_t &W = model.W;
	mat_t &H = model.H;

	// re-size nnz for ORIG
	if(param.solver_type == PU_SGD_ORIG) { // {{{
		for(int i = 0; i < training_set.rows; i++)
			training_set.nnz_row[i] = 0;
		for(int i = 0; i < training_set.cols; i++)
			training_set.nnz_col[i] = 0;
		for(size_t idx = 0; idx < training_set.nnz; idx++) {
			rate_t &r = training_set.allrates[idx];
			if(r.v > 0) {
				training_set.nnz_row[r.i]++;
				training_set.nnz_col[r.j]++;
			}
		}
	} // }}}

	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				vec_t &Wr = model.W[r];
				for(size_t t = 0; t < model.k; t++)
					Wr[t] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				vec_t &Hc = model.H[c];
				for(size_t t = 0; t < model.k; t++)
					Hc[t] = 0;
			}
	} // }}}

	sg_info_t info(training_set, model);

	std::vector<int> inner_cur_perm(innerB);
	std::vector<int> inner_perm(innerB); // column permutation
	std::vector<int> inner_offset(innerB); // row permutation
	for(int i = 0; i < innerB; i++)
		inner_perm[i] = inner_offset[i] = inner_cur_perm[i] = i;

	unsigned seed = 0;
	rng_t rng(seed);
	//std::srand(seed);

	omp_set_num_threads(param.threads);
	std::vector<rng_t > rng_set(param.threads);
#pragma omp parallel
	{
		int th = omp_get_thread_num();
		rng_set[th].seed(th+10);
	}

	double computetime = 0, tmpstart = 0;
	for(int iter = 1; iter <= maxiter; ++iter){
		double cur_lrate = lrate;
		if(iter > 1) info.slow_only = false;
		// Initialize permuation arrays {{{
		tmpstart = omp_get_wtime();
		rng.shuffle(inner_perm.begin(), inner_perm.end());
		rng.shuffle(inner_offset.begin(), inner_offset.end());
		//std::random_shuffle(inner_perm.begin(), inner_perm.end());
		//std::random_shuffle(inner_offset.begin(), inner_offset.end());
		computetime += omp_get_wtime() - tmpstart;
		// }}}

		// Computation {{{
		tmpstart = omp_get_wtime();
		// random stratum for inner blocks
		for(long i = 0; i < innerB; i++) {
			for(long ii = 0; ii < innerB; ii++)
				inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(dynamic,1) shared(training_set, param, model, info)
			for(long bi = 0; bi < innerB; bi++) {
				rng_t &local_rng = rng_set[omp_get_thread_num()];
				unsigned local_seed = bi * iter;
				long bj = inner_cur_perm[bi];
				long bid = training_set.get_bid(bi, bj);
				//sgd_core_orig(bid, training_set, param, model, cur_lrate, info, &local_seed, 1);
				//sgd_core_adgrad2(bid, training_set, param, model, cur_lrate, info, &local_seed, 1);
				//sgd_core_adgrad(bid, training_set, param, model, cur_lrate, info, &local_seed, 1);
				if(param.solver_type == SGD)
					sgd_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, 1);
				else if(param.solver_type == PU_SGD_ORIG)
					sgd_core_adgrad_weighted(bid, training_set, param, model, cur_lrate, info, local_rng, 1);
			}
		}
		computetime += omp_get_wtime() - tmpstart; 
		// }}}

		if(param.verbose) {
			double loss = 0.0;
			if(param.solver_type == PU_SGD_ORIG) 
				loss = compute_loss_orig(training_set, param, model);
			else 
				loss = compute_loss(training_set, param, model);
			double reg = compute_reg(training_set, param, model);

			printf("iter %d time %.10g loss %.10g train-rmse %.10g reg %.10g obj %.10g",
					iter, computetime, loss, sqrt(loss/training_set.nnz), reg, loss+lambda*reg);
		}

		if(param.do_predict && test_set.nnz!=0) {
			double test_loss = compute_loss(test_set, param, model);
			printf(" rmse %.10g", sqrt(test_loss/test_set.nnz));
		}
		if(param.verbose) puts("");
		fflush(stdout);
	}
	omp_set_num_threads(num_threads_old);
} // }}}



static inline void update_one_entry_omega_noraml(const rate_t &r, const pmf_parameter_t &param, pmf_model_t &model, 
		val_type lrate, val_type lambda_w, val_type lambda_h, val_type rho, sg_info_t &info, val_type amplified = val_type(1.0)) { // {{{
	size_t& k = model.k;
	size_t& k_slow = info.k_slow;
	val_type &invk_fast = info.invk_fast, &invk_slow = info.invk_slow;
	//val_type one_minus_rho = 1.0 - rho;
	lambda_w *= amplified;
	lambda_h *= amplified;

	val_type *Wi = model.W[r.i].data(), *Hj = model.H[r.j].data();
	val_type err = amplified*rho*(dot(Wi,Hj,k)-r.v);

	val_type lrate_w_slow=0, lrate_h_slow=0, grad_w_slow=0, grad_h_slow=0;
	val_type lrate_w_fast=0, lrate_h_fast=0, grad_w_fast=0, grad_h_fast=0;
	sg_info_t::grad_info_t &gw = info.grad_w[r.i], &gh = info.grad_h[r.j];

	if(k_slow > 0) {
		lrate_w_slow = lrate*rsqrt(gw.slow); 
		lrate_h_slow = lrate*rsqrt(gh.slow);
	}
	if(k_slow < k) {
		lrate_w_fast = lrate*rsqrt(gw.fast); 
		lrate_h_fast = lrate*rsqrt(gh.fast);
	}

	//  twin learners   {{{
	for(size_t t = 0; t < k_slow; t++) {
		val_type gw = err*Hj[t]+lambda_w*Wi[t];
		val_type gh = err*Wi[t]+lambda_h*Hj[t];
		grad_w_slow += gw*gw;
		grad_h_slow += gh*gh;
		Wi[t] -= lrate_w_slow*gw;
		Hj[t] -= lrate_h_slow*gh;
		if(0 && param.do_nmf) {
			if(Wi[t] < 0.0) Wi[t] = 0;
			if(Hj[t] < 0.0) Hj[t] = 0;
		}
	}
	if(!info.slow_only) // ???
		for(size_t t = k_slow; t < k; t++) {
			val_type gw = err*Hj[t]+lambda_w*Wi[t];
			val_type gh = err*Wi[t]+lambda_h*Hj[t];
			grad_w_fast += gw*gw;
			grad_h_fast += gh*gh;
			Wi[t] -= lrate_w_fast*gw;
			Hj[t] -= lrate_h_fast*gh;
			if(0 && param.do_nmf) {
				if(Wi[t] < 0.0) Wi[t] = 0;
				if(Hj[t] < 0.0) Hj[t] = 0;
			}
		}
	gw.slow += grad_w_slow*invk_slow;
	gh.slow += grad_h_slow*invk_slow;
	if(!info.slow_only) {
		gw.fast += grad_w_fast*invk_fast;
		gh.fast += grad_h_fast*invk_fast;
	}
	// }}}
} // }}}

static inline void update_one_entry_omega(const rate_t &r, const pmf_parameter_t &param, pmf_model_t &model, val_type lrate, val_type lambda_w, val_type lambda_h, sg_info_t &info, val_type amplified = val_type(1.0)) { // {{{
	size_t& k = model.k;
	size_t& k_slow = info.k_slow;
	val_type &invk_fast = info.invk_fast, &invk_slow = info.invk_slow;
	val_type one_minus_rho = 1.0 - param.rho;
	lambda_w *= amplified;
	lambda_h *= amplified;

	val_type *Wi = model.W[r.i].data(), *Hj = model.H[r.j].data();
	val_type err = amplified*(one_minus_rho*dot(Wi,Hj,k)-r.v);

	val_type lrate_w_slow=0, lrate_h_slow=0, grad_w_slow=0, grad_h_slow=0;
	val_type lrate_w_fast=0, lrate_h_fast=0, grad_w_fast=0, grad_h_fast=0;
	sg_info_t::grad_info_t &gw = info.grad_w[r.i], &gh = info.grad_h[r.j];

	if(k_slow > 0) {
		lrate_w_slow = lrate*rsqrt(gw.slow); 
		lrate_h_slow = lrate*rsqrt(gh.slow);
	}
	if(k_slow < k) {
		lrate_w_fast = lrate*rsqrt(gw.fast); 
		lrate_h_fast = lrate*rsqrt(gh.fast);
	}

	//  twin learners   {{{
	for(size_t t = 0; t < k_slow; t++) {
		val_type gw = err*Hj[t]+lambda_w*Wi[t];
		val_type gh = err*Wi[t]+lambda_h*Hj[t];
		grad_w_slow += gw*gw;
		grad_h_slow += gh*gh;
		Wi[t] -= lrate_w_slow*gw;
		Hj[t] -= lrate_h_slow*gh;
		if(param.do_nmf) {
			if(Wi[t] < 0.0) Wi[t] = 0;
			if(Hj[t] < 0.0) Hj[t] = 0;
		}
	}
	if(!info.slow_only) // ???
		for(size_t t = k_slow; t < k; t++) {
			val_type gw = err*Hj[t]+lambda_w*Wi[t];
			val_type gh = err*Wi[t]+lambda_h*Hj[t];
			grad_w_fast += gw*gw;
			grad_h_fast += gh*gh;
			Wi[t] -= lrate_w_fast*gw;
			Hj[t] -= lrate_h_fast*gh;
			if(param.do_nmf) {
				if(Wi[t] < 0.0) Wi[t] = 0;
				if(Hj[t] < 0.0) Hj[t] = 0;
			}
		}
	gw.slow += grad_w_slow*invk_slow;
	gh.slow += grad_h_slow*invk_slow;
	if(!info.slow_only) {
		gw.fast += grad_w_fast*invk_fast;
		gh.fast += grad_h_fast*invk_fast;
	}
	// }}}
} // }}}

static inline void update_one_entry_zero(const rate_t &r, const pmf_parameter_t &param, pmf_model_t &model, val_type lrate, val_type lambda_w, val_type lambda_h, sg_info_t &info, val_type amplified = val_type(1.0)) { // {{{
	size_t& k = model.k;
	size_t& k_slow = info.k_slow;
	val_type &invk_fast = info.invk_fast, &invk_slow = info.invk_slow;
	lambda_w *= amplified; 
	lambda_h *= amplified; 

	val_type *Wi = model.W[r.i].data(), *Hj = model.H[r.j].data();
	val_type err = amplified*param.rho*dot(Wi,Hj,k);

	val_type lrate_w_slow=0, lrate_h_slow=0, grad_w_slow=0, grad_h_slow=0;
	val_type lrate_w_fast=0, lrate_h_fast=0, grad_w_fast=0, grad_h_fast=0;
	sg_info_t::grad_info_t &gw = info.grad_w[r.i], &gh = info.grad_h[r.j];

	if(k_slow > 0)
		lrate_w_slow = lrate*rsqrt(gw.slow); lrate_h_slow = lrate*rsqrt(gh.slow);
	if(k_slow < k)
		lrate_w_fast = lrate*rsqrt(gw.fast); lrate_h_fast = lrate*rsqrt(gh.fast);

	//  twin learners   {{{
	for(size_t t = 0; t < k_slow; t++) {
		val_type gw = err*Hj[t]+lambda_w*Wi[t];
		val_type gh = err*Wi[t]+lambda_h*Hj[t];
		grad_w_slow += gw*gw;
		grad_h_slow += gh*gh;
		Wi[t] -= lrate_w_slow*gw;
		Hj[t] -= lrate_h_slow*gh;
		if(param.do_nmf) {
			if(Wi[t] < 0.0) Wi[t] = 0;
			if(Hj[t] < 0.0) Hj[t] = 0;
		}
	}
	if(!info.slow_only) // ???
		for(size_t t = k_slow; t < k; t++) {
			val_type gw = err*Hj[t]+lambda_w*Wi[t];
			val_type gh = err*Wi[t]+lambda_h*Hj[t];
			grad_w_fast += gw*gw;
			grad_h_fast += gh*gh;
			Wi[t] -= lrate_w_fast*gw;
			Hj[t] -= lrate_h_fast*gh;
			if(param.do_nmf) {
				if(Wi[t] < 0.0) Wi[t] = 0;
				if(Hj[t] < 0.0) Hj[t] = 0;
			}
		}
	gw.slow += grad_w_slow*invk_slow;
	gh.slow += grad_h_slow*invk_slow;
	if(!info.slow_only) {
		gw.fast += grad_w_fast*invk_fast;
		gh.fast += grad_h_fast*invk_fast;
	}
	// }}}
} // }}}

static void sgd_pu0_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = param.rho;
	val_type one_minus_rho = 1.0 - rho;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = blocknnz*maxiter;
	size_t postive_updates = (size_t)(total_updates *(double) blocknnz/(blocknnz + rho*sub_rows*sub_cols));
	size_t unseen_updates = total_updates - postive_updates;

	// unseen_updates
	for(size_t idx = 0; idx <unseen_updates; idx++) { 
		rate_t r;
		r.i = start_row + rng.randrange(sub_rows);
		r.j = start_col + rng.randrange(sub_cols);
		r.v = 0;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		val_type zero_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		val_type zero_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
	} 
	// postive updates
	for(size_t idx = 0; idx < postive_updates; idx++) { 
		const rate_t &r = block[info.counts[bid]%blocknnz];
		info.counts[bid]++;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
	} 
} // }}}

// first omega next unseen
static void sgd_pu1_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = param.rho;
	val_type one_minus_rho = 1.0 - rho;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = blocknnz*(1+maxiter);
	size_t postive_updates = total_updates*1.0/(1.0+maxiter);
	size_t unseen_updates = total_updates - postive_updates;

	// postive updates
	for(size_t idx = 0; idx < postive_updates; idx++) { 
		const rate_t &r = block[info.counts[bid]%blocknnz];
		info.counts[bid]++;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
	}
	// unseen_updates
	val_type amplified = (double)sub_rows*sub_cols*rho/(blocknnz*maxiter);
	for(size_t idx = 0; idx <unseen_updates; idx++) { 
		rate_t r;
		r.i = start_row + rng.randrange(sub_rows);
		r.j = start_col + rng.randrange(sub_cols);
		r.v = 0;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		val_type zero_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		val_type zero_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
	} 
} // }}}

// rand sampling with shifted loss + reg on both loss terms
static void sgd_pu2_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = (val_type)param.rho;

	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	//size_t total_updates = blocknnz*(maxiter);
	size_t total_updates = sub_rows*sub_cols;
	double threshold = (double)blocknnz/(blocknnz + sub_rows*sub_cols);

	for(size_t idx = 0; idx < total_updates; idx++) {
		double sample = rng.uniform(0.0, 1.0);
		if(sample < threshold) { // postive updates
			//const rate_t &r = block[info.counts[bid]%blocknnz];
			const rate_t &r = block[rng.randrange(blocknnz)];
			info.counts[bid]++;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
			val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
			update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
		} else { // unseen updates
			rate_t r;
			r.i = start_row+rng.randrange(sub_rows);
			r.j = start_col+rng.randrange(sub_cols);
			r.v = 0;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type zero_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
			val_type zero_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
			update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
		}
	}
} // }}}

// gd + sgd
static void sgd_pu3_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = param.rho;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t postive_updates = blocknnz*(maxiter);

	// postive updates
	for(size_t idx = 0; idx < postive_updates; idx++) { 
		const rate_t &r = block[info.counts[bid]%blocknnz];
		info.counts[bid]++;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		//val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		//val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		//val_type omega_lambda_w = lambda*nz_i/(1.+nz_i);
		//val_type omega_lambda_h = lambda*nz_j/(1.+nz_j);
		val_type omega_lambda_w = lambda; 
		val_type omega_lambda_h = lambda; 
		//update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
		update_one_entry_omega_noraml(r, param, model, lrate, omega_lambda_w, omega_lambda_h, 1.0, info, 1.0);
	}

	// unseen_updates
	int tid = omp_get_thread_num();
	val_type *subW = model.W[start_row].data();
	val_type *subH = model.H[start_col].data();
	val_type *WTW = info.WTW[tid].data();
	val_type *HTH = info.HTH[tid].data();
	val_type *WHTH = info.WHTH[tid].data();
	val_type *HWTW = info.HWTW[tid].data();

	size_t k = model.k;
	size_t rows = training_set.rows;
	size_t cols = training_set.cols;

	val_type lrate_gd = lrate; //(1+info.tt); // /sqrt(sub_rows*sub_cols);

	for(int subiter = 0; subiter < 1; subiter++) {
		doHTH(subH, HTH, sub_cols, k);
		dmat_x_dmat(subW, HTH, WHTH, sub_rows, k, k);
		for(int i = 0; i < sub_rows; i++) {
			size_t nz_i = training_set.nnz_of_row(start_row+i);
			val_type lambda_w = lambda*nz_i*sub_cols/(nz_i+cols);
			//lambda_w = lambda*nz_i/(nz_i+1.0);
			//lambda_w = lambda*nz_i;
			lambda_w = 0;
			for(int t = 0; t < k; t++)
				WHTH[i*k+t] = rho*WHTH[i*k+t]+lambda_w*subW[i*k+t];
		}
		do_axpy(-lrate_gd, WHTH, subW, sub_rows*k);

		doHTH(subW, WTW, sub_rows, k);
		dmat_x_dmat(subH, WTW, HWTW, sub_cols, k, k);
		for(int j = 0; j < sub_cols; j++) {
			size_t nz_j = training_set.nnz_of_col(start_col+j);
			val_type lambda_h = lambda*nz_j*sub_rows/(nz_j+rows);
			//lambda_h = lambda*nz_j/(nz_j+1.0);
			//lambda_h = lambda*nz_j;
			lambda_h = 0;
			for(int t = 0; t < k; t++)
				HWTW[j*k+t] = rho*HWTW[j*k+t]+lambda_h*subH[j*k+t];
		}
		do_axpy(-lrate_gd, HWTW, subH, sub_cols*k);
	}

} // }}}

// naive implementation with binary search  OK
static void sgd_pu4_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;

	size_t blocknnz = block.size();

	val_type rho = (val_type)param.rho, alpha = (val_type)param.alpha, amplified = 1.0/alpha;
	val_type one_minus_rho = 1.0 - rho;

	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = blocknnz*(maxiter);

	val_type zero_lambda_w = param.lambda/training_set.cols;
	val_type zero_lambda_h = param.lambda/training_set.rows;
	val_type omega_lambda_w = param.lambda/training_set.cols;
	val_type omega_lambda_h = param.lambda/training_set.rows;
	val_type zero_rho = param.rho, omega_rho = 1.0;

	//for(size_t idx = 0; idx < total_updates; idx++) {
	for(int j = 0; j < sub_cols; j++) 
		for(int i = 0; i < sub_rows; i++) {
			rate_t r; // = block[idx%block.size()];
			//r.i = start_row+rng.randrange(sub_rows);
			//r.j = start_col+rng.randrange(sub_cols);
			r.i = start_row + i;
			r.j = start_col + j;
			r.v = 0;

			{
				const rate_t *ptr = block.find(r);
				if(ptr != NULL) 
					r.v = ptr->v;

				if(r.v != 0)  {
					update_one_entry_omega_noraml(r, param, model, lrate, 
							training_set.nnz_of_row(r.i)*omega_lambda_w, training_set.nnz_of_col(r.j)*omega_lambda_h, omega_rho, info);
					//omega_lambda_w, omega_lambda_h, omega_rho, info);
				} else  {
					update_one_entry_omega_noraml(r, param, model, lrate, 
							training_set.nnz_of_row(r.i)*zero_lambda_w, training_set.nnz_of_col(r.j)*zero_lambda_h, zero_rho, info);
					//training_set.nnz_of_row(r.i)*zero_lambda_w, training_set.nnz_of_col(r.j)*zero_lambda_h, zero_rho, info);
				}
			}
		}
} // }}}

// cyclic go through  OK
static void sgd_pu5_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;

	/*
	for(size_t idx = 0; idx < blocknnz; idx++) {
		const rate_t &r = block[idx];
		info.counts[bid]++;

		size_t nz_i = training_set.nnz_of_row(r.i);
		size_t nz_j = training_set.nnz_of_col(r.j);
		val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
		val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
		update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
	}
	*/

	for(int ii = 0; ii < sub_rows; ii++) {
		for(int jj = 0; jj < sub_cols; jj++) {
			rate_t r;
			r.i = start_row+ii;
			r.j = start_col+jj; 

			const rate_t *ptr = block.find(r);
			if(ptr != NULL)  {
				r.v = ptr->v;
				size_t nz_i = training_set.nnz_of_row(r.i);
				size_t nz_j = training_set.nnz_of_col(r.j);
				val_type omega_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
				val_type omega_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
				update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);

			}

			r.v = 0;
			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type zero_lambda_w = lambda*nz_i/(training_set.cols+nz_i);
			val_type zero_lambda_h = lambda*nz_j/(training_set.rows+nz_j);
			update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
		}
	}
} // }}}

// rand sampling with binary search
// naive implementation with binary search  OK
static void sgd_pu6_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;

	size_t blocknnz = block.size();

	val_type rho = (val_type)param.rho, alpha = (val_type)param.alpha, amplified = 1.0/alpha;
	val_type one_minus_rho = 1.0 - rho;

	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = sub_rows*sub_cols;

	val_type zero_lambda_w = param.lambda/training_set.cols;
	val_type zero_lambda_h = param.lambda/training_set.rows;
	val_type omega_lambda_w = param.lambda/training_set.cols;
	val_type omega_lambda_h = param.lambda/training_set.rows;
	val_type zero_rho = param.rho, omega_rho = 1.0;

	for(size_t idx = 0; idx < total_updates; idx++) {
		rate_t r; // = block[idx%block.size()];
		r.i = start_row+rng.randrange(sub_rows);
		r.j = start_col+rng.randrange(sub_cols);
		r.v = 0;

		{
			const rate_t *ptr = block.find(r);
			if(ptr != NULL) 
				r.v = ptr->v;

			if(r.v != 0)  {
				update_one_entry_omega_noraml(r, param, model, lrate, 
						training_set.nnz_of_row(r.i)*omega_lambda_w, training_set.nnz_of_col(r.j)*omega_lambda_h, omega_rho, info);
				//omega_lambda_w, omega_lambda_h, omega_rho, info);
			} else  {
				update_one_entry_omega_noraml(r, param, model, lrate, 
						training_set.nnz_of_row(r.i)*zero_lambda_w, training_set.nnz_of_col(r.j)*zero_lambda_h, zero_rho, info);
				//training_set.nnz_of_row(r.i)*zero_lambda_w, training_set.nnz_of_col(r.j)*zero_lambda_h, zero_rho, info);
			}
		}
	}
} // }}}

// rand sampling with shifted loss + reg on \ell^+ only
static void sgd_pu7_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = (val_type)param.rho;

	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	//size_t total_updates = blocknnz*(maxiter);
	size_t total_updates = sub_rows*sub_cols;
	double threshold = (double)blocknnz/(blocknnz + sub_rows*sub_cols);

	for(size_t idx = 0; idx < total_updates; idx++) {
		double sample = rng.uniform(0.0, 1.0);
		if(sample < threshold) { // postive updates
			//const rate_t &r = block[info.counts[bid]%blocknnz];
			const rate_t &r = block[rng.randrange(blocknnz)];
			info.counts[bid]++;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type omega_lambda_w = lambda;
			val_type omega_lambda_h = lambda;
			update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
		} else { // unseen updates
			rate_t r;
			r.i = start_row+rng.randrange(sub_rows);
			r.j = start_col+rng.randrange(sub_cols);
			r.v = 0;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type zero_lambda_w = 0;
			val_type zero_lambda_h = 0; 
			update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
		}
	}
} // }}}

// rand sampling with shifted loss + reg on \ell^- only
static void sgd_pu8_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, rng_t &rng, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	size_t blocknnz = block.size();

	val_type lambda = param.lambda;
	val_type rho = (val_type)param.rho;

	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	//size_t total_updates = blocknnz*(maxiter);
	size_t total_updates = sub_rows*sub_cols;
	double threshold = (double)blocknnz/(blocknnz + sub_rows*sub_cols);

	for(size_t idx = 0; idx < total_updates; idx++) {
		double sample = rng.uniform(0.0, 1.0);
		if(sample < threshold) { // postive updates
			//const rate_t &r = block[info.counts[bid]%blocknnz];
			const rate_t &r = block[rng.randrange(blocknnz)];
			info.counts[bid]++;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type omega_lambda_w = 0; //lambda*nz_i/(training_set.cols+nz_i);
			val_type omega_lambda_h = 0; //lambda*nz_j/(training_set.rows+nz_j);
			update_one_entry_omega(r, param, model, lrate, omega_lambda_w, omega_lambda_h, info);
		} else { // unseen updates
			rate_t r;
			r.i = start_row+rng.randrange(sub_rows);
			r.j = start_col+rng.randrange(sub_cols);
			r.v = 0;

			size_t nz_i = training_set.nnz_of_row(r.i);
			size_t nz_j = training_set.nnz_of_col(r.j);
			val_type zero_lambda_w = lambda*nz_i/(training_set.cols);
			val_type zero_lambda_h = lambda*nz_j/(training_set.rows);
			update_one_entry_zero(r, param, model, lrate, zero_lambda_w, zero_lambda_h, info);
		}
	}
} // }}}

#ifdef old
static void sgd_pu0_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, unsigned *seed, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;

	size_t blocknnz = block.size();

	val_type rho = param.rho;
	val_type one_minus_rho = 1.0 - rho;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = blocknnz*maxiter;
	size_t postive_updates = (size_t)(total_updates *(double) blocknnz/(blocknnz + rho*sub_rows*sub_cols));
	size_t unseen_updates = total_updates - postive_updates;

	// unseen_updates
	for(size_t idx = 0; idx <unseen_updates; idx++) { 
		rate_t r;
		r.i = start_row + rand_r(seed)%sub_rows;
		r.j = start_col + rand_r(seed)%sub_cols;
		r.v = 0;
		update_one_entry_zero(r, param, model, lrate, info);
	} 
	// postive updates
	for(size_t idx = 0; idx < postive_updates; idx++) { 
		const rate_t &r = block[info.counts[bid]%blocknnz];
		info.counts[bid]++;
		update_one_entry_omega(r, param, model, lrate, info);
	} 
} // }}}

static void sgd_pu1_core_adgrad(int bid, const blocks_t &training_set, const pmf_parameter_t &param, pmf_model_t &model,
		const val_type &lrate, sg_info_t& info, unsigned *seed, int maxiter=1) { //{{{
	const block_t& block = training_set[bid];
	mat_t &W = model.W, &H = model.H;
	val_type lambda = param.lambda;

	size_t blocknnz = block.size();

	val_type rho = param.rho;
	val_type one_minus_rho = 1.0 - rho;
	size_t start_row = block.start_row, start_col = block.start_col;
	size_t sub_rows = block.sub_rows, sub_cols = block.sub_cols;
	size_t total_updates = blocknnz*(1+maxiter);
	size_t postive_updates = total_updates*1.0/(1.0+maxiter);
	size_t unseen_updates = total_updates - postive_updates;

	// postive updates
	for(size_t idx = 0; idx < postive_updates; idx++) { 
		const rate_t &r = block[info.counts[bid]%blocknnz];
		info.counts[bid]++;
		update_one_entry_omega(r, param, model, lrate, info);
	}
	// unseen_updates
	val_type amplified = (double)sub_rows*sub_cols*rho/(blocknnz*maxiter);
	for(size_t idx = 0; idx <unseen_updates; idx++) { 
		rate_t r;
		r.i = start_row + rand_r(seed)%sub_rows;
		r.j = start_col + rand_r(seed)%sub_cols;
		r.v = 0;
		update_one_entry_zero(r, param, model, lrate, info, amplified);

	} 
} // }}}
#endif


void sgd_pu(blocks_t &training_set, blocks_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	if(model.major_type != pmf_model_t::ROWMAJOR) {
		fprintf(stderr, "SGD requires pmf_model_t::ROWMAJOR model\n");
		return;
	}

	int maxiter = param.maxiter;
	int num_threads_old = omp_get_num_threads();
	size_t k = param.k;
	double lambda = param.lambda;
	double lrate = param.eta0;
	int innerB = param.nr_blocks;

	if(innerB <= 0) innerB = 4*param.threads;

	bool row_major = (training_set.rows > training_set.cols)? true : false;
	training_set.set_blocks(innerB, row_major);
	test_set.set_blocks(innerB, row_major);
	mat_t &W = model.W;
	mat_t &H = model.H;

	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				vec_t &Wr = model.W[r];
				for(size_t t = 0; t < model.k; t++)
					Wr[t] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				vec_t &Hc = model.H[c];
				for(size_t t = 0; t < model.k; t++)
					Hc[t] = 0;
			}
	} // }}}

	sg_info_t info(training_set, model, param.threads);

	std::vector<int> inner_cur_perm(innerB);
	std::vector<int> inner_perm(innerB); // column permutation
	std::vector<int> inner_offset(innerB); // row permutation
	for(int i = 0; i < innerB; i++)
		inner_perm[i] = inner_offset[i] = inner_cur_perm[i] = i;

	unsigned seed = 0;

#ifdef old
	std::srand(seed);
#endif
	rng_t rng(seed);

	omp_set_num_threads(param.threads);
	std::vector<rng_t > rng_set(param.threads);
#pragma omp parallel
	{
		int th = omp_get_thread_num();
		rng_set[th].seed(th+10);
	}

	double computetime = 0, tmpstart = 0;
	double cur_lrate = lrate, oldobj = 0;
	for(int iter = 1; iter <= maxiter; ++iter){
		if(iter > 1) info.slow_only = false;
		// Initialize permuation arrays {{{
		tmpstart = omp_get_wtime();
#ifdef old
		std::random_shuffle(inner_perm.begin(), inner_perm.end());
		std::random_shuffle(inner_offset.begin(), inner_offset.end());
#else
		rng.shuffle(inner_perm.begin(), inner_perm.end());
		rng.shuffle(inner_offset.begin(), inner_offset.end());
#endif
		computetime += omp_get_wtime() - tmpstart;
		// }}}

		// Computation {{{
		tmpstart = omp_get_wtime();
		info.tt = iter;
		// random stratum for inner blocks
		for(long i = 0; i < innerB; i++) {
			for(long ii = 0; ii < innerB; ii++)
				inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(dynamic,1) shared(training_set, param, model, info)
			for(long bi = 0; bi < innerB; bi++) {
				rng_t &local_rng = rng_set[omp_get_thread_num()];
				unsigned local_seed = bi * iter;
				long bj = inner_cur_perm[bi];
				long bid = training_set.get_bid(bi, bj);
#ifdef old
				if(param.pu_type == PU0)
					sgd_pu0_core_adgrad(bid, training_set, param, model, cur_lrate, info, &local_seed, param.maxinneriter);
				else if(param.pu_type == PU1)
					sgd_pu1_core_adgrad(bid, training_set, param, model, cur_lrate, info, &local_seed, param.maxinneriter);
#else 
				if(param.pu_type == PU0)
					sgd_pu0_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU1)
					sgd_pu1_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU2)
					sgd_pu2_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU3)
					sgd_pu3_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU4)
					sgd_pu4_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU5)
					sgd_pu5_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU6)
					sgd_pu6_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU7)
					sgd_pu7_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU8)
					sgd_pu8_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
#endif
			}
		}
		computetime += omp_get_wtime() - tmpstart; // }}}

	double loss_omega, loss_zero;
	double loss = compute_pu_loss(training_set, param, model, &loss_omega, &loss_zero); 
	double reg = compute_reg(training_set, param, model);
	double obj = loss+lambda*reg;
	oldobj = obj;

	if(param.verbose) {
		printf("iter %d time %.10g loss %.10g omega %.10g zero %.10g train-rmse %.10g reg %.10g obj %.10g",
				iter, computetime, loss, loss_omega, loss_zero, sqrt(loss/training_set.nnz), reg, loss+lambda*reg);
	}

	if(param.do_predict && test_set.nnz!=0) {
		double test_loss = compute_loss(test_set, param, model);
		printf(" rmse %.10g", sqrt(test_loss/test_set.nnz));
	}
	if(param.verbose) puts("");
	fflush(stdout);
	}
	omp_set_num_threads(num_threads_old);
} // }}}


void gd_pu(blocks_t &training_set, blocks_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	if(model.major_type != pmf_model_t::ROWMAJOR) {
		fprintf(stderr, "SGD requires pmf_model_t::ROWMAJOR model\n");
		return;
	}

	int maxiter = param.maxiter;
	int num_threads_old = omp_get_num_threads();
	size_t k = param.k;
	double lambda = param.lambda;
	double lrate = param.eta0;
	int innerB = param.nr_blocks;

	if(innerB <= 0) innerB = 4*param.threads;

	bool row_major = (training_set.rows > training_set.cols)? true : false;
	training_set.set_blocks(innerB, row_major);
	test_set.set_blocks(innerB, row_major);
	mat_t &W = model.W;
	mat_t &H = model.H;

	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				vec_t &Wr = model.W[r];
				for(size_t t = 0; t < model.k; t++)
					Wr[t] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				vec_t &Hc = model.H[c];
				for(size_t t = 0; t < model.k; t++)
					Hc[t] = 0;
			}
	} // }}}

	sg_info_t info(training_set, model, param.threads);

	std::vector<int> inner_cur_perm(innerB);
	std::vector<int> inner_perm(innerB); // column permutation
	std::vector<int> inner_offset(innerB); // row permutation
	for(int i = 0; i < innerB; i++)
		inner_perm[i] = inner_offset[i] = inner_cur_perm[i] = i;

	unsigned seed = 0;

#ifdef old
	std::srand(seed);
#endif
	rng_t rng(seed);

	omp_set_num_threads(param.threads);
	std::vector<rng_t > rng_set(param.threads);
#pragma omp parallel
	{
		int th = omp_get_thread_num();
		rng_set[th].seed(th+10);
	}

	double computetime = 0, tmpstart = 0;
	double cur_lrate = lrate, oldobj = 0;
	for(int iter = 1; iter <= maxiter; ++iter){
		if(iter > 1) info.slow_only = false;
		// Initialize permuation arrays {{{
		tmpstart = omp_get_wtime();
#ifdef old
		std::random_shuffle(inner_perm.begin(), inner_perm.end());
		std::random_shuffle(inner_offset.begin(), inner_offset.end());
#else
		rng.shuffle(inner_perm.begin(), inner_perm.end());
		rng.shuffle(inner_offset.begin(), inner_offset.end());
#endif
		computetime += omp_get_wtime() - tmpstart;
		// }}}

		// Computation {{{
		tmpstart = omp_get_wtime();
		// random stratum for inner blocks
		for(long i = 0; i < innerB; i++) {
			for(long ii = 0; ii < innerB; ii++)
				inner_cur_perm[inner_perm[ii]] = (ii+inner_offset[i])%innerB;
#pragma omp parallel for schedule(dynamic,1) shared(training_set, param, model, info)
			for(long bi = 0; bi < innerB; bi++) {
				rng_t &local_rng = rng_set[omp_get_thread_num()];
				unsigned local_seed = bi * iter;
				long bj = inner_cur_perm[bi];
				long bid = training_set.get_bid(bi, bj);
#ifdef old
				if(param.pu_type == PU0)
					sgd_pu0_core_adgrad(bid, training_set, param, model, cur_lrate, info, &local_seed, param.maxinneriter);
				else if(param.pu_type == PU1)
					sgd_pu1_core_adgrad(bid, training_set, param, model, cur_lrate, info, &local_seed, param.maxinneriter);
#else 
				if(param.pu_type == PU0)
					sgd_pu0_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU1)
					sgd_pu1_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU2)
					sgd_pu2_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
				else if(param.pu_type == PU3)
					sgd_pu3_core_adgrad(bid, training_set, param, model, cur_lrate, info, local_rng, param.maxinneriter);
#endif
			}
		}
		computetime += omp_get_wtime() - tmpstart; // }}}

	double loss_omega, loss_zero;
	double loss = compute_pu_loss(training_set, param, model, &loss_omega, &loss_zero); 
	double reg = compute_reg(training_set, param, model);
	double obj = loss+lambda*reg;
	if(iter > 3) {
		if(obj < oldobj) {
			cur_lrate *= 1.05;
			printf("cur_lrate increases %g\n", cur_lrate);
		} else {
			cur_lrate *= 0.5;
			printf("cur_lrate decreases %g\n", cur_lrate);
		}
	}
	oldobj = obj;

	if(param.verbose) {
		printf("iter %d time %.10g loss_omega %.10g loss_zero %.10g loss %.10g train-rmse %.10g reg %.10g obj %.10g",
				iter, computetime, loss_omega, loss_zero, loss, sqrt(loss/training_set.nnz), reg, loss+lambda*reg);
	}

	if(param.do_predict && test_set.nnz!=0) {
		double test_loss = compute_loss(test_set, param, model);
		printf(" rmse %.10g", sqrt(test_loss/test_set.nnz));
	}
	if(param.verbose) puts("");
	fflush(stdout);
	}
	omp_set_num_threads(num_threads_old);
} // }}}
