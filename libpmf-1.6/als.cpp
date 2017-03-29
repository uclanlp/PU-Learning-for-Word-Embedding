#include "pmf.h"
#define kind dynamic,64

// solve an ALS subproblem wrt j-th column of R, the result is stored in y
//    if y == NULL, the result is stored in Hj directly
//static inline void updateOne(vec_t &Hj, smat_t &R, const int j, const mat_t& W, const val_type lambda, 
static inline void updateOne(vec_t &Hj, smat_t &R, const int j, const mat_t& W, const val_type lambda, 
		val_type *Hessian, val_type *y = NULL) { // {{{
	if(R.nnz_of_col(j)<=0) return;

	size_t k = Hj.size();
	if(y==NULL) y = &Hj[0];

	memset(Hessian, 0, sizeof(val_type)*k*k);
	memset(y, 0, sizeof(val_type)*k);

	// Construct k*k Hessian and k*1 y
	for(size_t idx = R.col_ptr[j]; idx != R.col_ptr[j+1]; ++idx){
		const vec_t &Wi = W[R.row_idx[idx]];
		for(size_t s = 0; s < k; ++s) {
			y[s] += R.val[idx]*Wi[s];
			for(size_t t = s; t < k; ++t)
				Hessian[s*k+t] += Wi[s]*Wi[t];
		}
	}
	for(size_t s = 0; s < k; ++s) {
		for(size_t t = 0; t < s; ++t)
			Hessian[s*k+t] = Hessian[t*k+s];
		Hessian[s*k+s] += lambda;
	}

	ls_solve_chol(Hessian, y, k);
} // }}}

static double compute_loss(const smat_t &R, const pmf_parameter_t &param, pmf_model_t &model) { // {{{
	const mat_t &W = model.W, &H = model.H;
	double loss = 0.0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
	for(long c = 0; c < R.cols; c++) {
		val_type loss_inner = 0.0;
		const vec_t &Hj = H[c];
		for(size_t idx = R.col_ptr[c]; idx != R.col_ptr[c+1]; ++idx) {
			size_t r = R.row_idx[idx];
			val_type sum = -R.val[idx];
			const vec_t &Wi = W[r];
			for(size_t t = 0; t < model.k; ++t)
				sum += Wi[t] * Hj[t];
			loss_inner += sum*sum;
		}
		loss += loss_inner;
	}
	return loss;
} // }}}

static double compute_pu_loss(const smat_t &R, const pmf_parameter_t &param, pmf_model_t &model, double *loss_omega, double *loss_zero) { // {{{
	const mat_t &W = model.W, &H = model.H;
	vec_t omega_parts(param.threads), zero_parts(param.threads);
	memset(omega_parts.data(), 0, param.threads*sizeof(val_type));
	memset(zero_parts.data(), 0, param.threads*sizeof(val_type));

#pragma omp parallel for schedule(kind) 
	for(long c = 0; c < R.cols; c++) {
		val_type omega_part = val_type(0.0), zero_part = val_type(0.0);

		const vec_t &Hj = H[c];
		for(size_t idx = R.col_ptr[c]; idx != R.col_ptr[c+1]; ++idx) {
			size_t r = R.row_idx[idx];
			val_type sum = 0.0; 
			const vec_t &Wi = W[r];
			for(size_t t = 0; t < model.k; ++t)
				sum += Wi[t] * Hj[t];
			zero_part -= sum*sum;
			sum -= R.val[idx];
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

static double compute_reg(const smat_t &R, const pmf_parameter_t &param, pmf_model_t &model) { // {{{
	double norm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:norm)
	for(long r = 0; r < model.rows; r++) {
		const vec_t &Wi = model.W[r];
		for(size_t t = 0; t < model.k; t++)
			norm += Wi[t]*Wi[t]*R.nnz_of_row(r);
	}
#pragma omp parallel for schedule(static) reduction(+:norm)
	for(long c = 0; c < model.cols; c++) {
		const vec_t &Hj = model.H[c];
		for(size_t t = 0; t < model.k; t++)
			norm += Hj[t]*Hj[t]*R.nnz_of_col(c);
	}
	return norm;
} // }}}

// Alternating Least Squares for Matrix Factorization
void als(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	if(model.major_type != pmf_model_t::ROWMAJOR) {
		fprintf(stderr, "ALS requires pmf_model_t::ROWMAJOR model\n");
		return;
	}

	int num_threads_old = omp_get_num_threads();
	int maxiter = param.maxiter;
	val_type lambda = param.lambda;
	omp_set_num_threads(param.threads);
	
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

	// Create transpose view of R
	smat_t &R = training_set, Rt = R.transpose();
	smat_t &testR = test_set, testRt = testR.transpose();
	mat_t &W = model.W, &H = model.H;

	std::vector<vec_t> Hessian_set(param.threads, vec_t(model.k*model.k));
	std::vector<vec_t> &AA = Hessian_set;
	//std::vector<std::vector<double> > Hessian_set(param.threads, std::vector<double>(model.k*model.k));

	double Wtime = 0, Htime = 0, start = 0;
	for(int iter = 1; iter <= maxiter; ++iter) {
		// Update H
		start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(R,H,W,Hessian_set)
		for(long c = 0; c < R.cols; ++c) {
			int tid = omp_get_thread_num();
			updateOne(H[c], R, c, W, lambda*R.nnz_of_col(c), &Hessian_set[tid][0]);
		}
		Htime += omp_get_wtime() - start;
		// Update W
		start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(R,Rt,H,W,Hessian_set)
		for(long c = 0; c < Rt.cols; ++c) {
			int tid = omp_get_thread_num();
			updateOne(W[c], Rt, c, H, lambda*Rt.nnz_of_col(c), &Hessian_set[tid][0]);
		}
		Wtime += omp_get_wtime() - start;

		if(param.verbose) {
			double loss = compute_loss(R, param, model);
			double reg = compute_reg(R, param, model);

			printf("iter %d time %.10g loss %.10g train-rmse %.10g reg %.10g obj %.10g", 
					iter, Htime+Wtime, loss, sqrt(loss/R.nnz), reg, loss+lambda*reg);
		}

		if(param.do_predict && testR.nnz!=0) {
			double test_loss = compute_loss(testR, param, model);
			printf(" rmse %.10g", sqrt(test_loss/testR.nnz)); 
		}
		if(param.verbose) puts("");
		fflush(stdout);
	}
	omp_set_num_threads(num_threads_old);

} // }}}


static void compute_rhoWTW(const mat_t &W, vec_t &rhoWTW, size_t k, val_type rho) { // {{{
	memset(&rhoWTW[0], 0, sizeof(val_type)*k*k);
	for(size_t i = 0; i < W.size(); i++) {
		const vec_t& Wi = W[i];
		for(int s = 0; s < k; ++s)
			for(int t = s; t < k; ++t)
				rhoWTW[s*k+t] += Wi[t]*Wi[s];
	}
	for(int s = 0; s < k; ++s)
		for(int t = 0; t <= s; t++) {
			val_type tmp = (rhoWTW[t*k+s] *= rho);
			rhoWTW[s*k+t] = tmp;
		}
} // }}}

static inline void updateOne_pu(vec_t& Hj, smat_t &R, const int j, const mat_t& W, const val_type lambda, const val_type rho,
		val_type *rhoWTW, val_type *Hessian, val_type *y = NULL) { // {{{
	if(R.nnz_of_col(j)<=0) return;

	size_t k = Hj.size();
	if(y==NULL) y = &Hj[0];

	const val_type one_minus_rho = 1-rho;

	memcpy(Hessian, rhoWTW, sizeof(val_type)*k*k);
	memset(y, 0, sizeof(val_type)*k);

	// Construct k*k Hessian and k*1 y
	for(size_t idx = R.col_ptr[j]; idx != R.col_ptr[j+1]; ++idx){
		const vec_t &Wi = W[R.row_idx[idx]];
		for(size_t s = 0; s < k; ++s) {
			y[s] += R.val[idx]*Wi[s];
			for(size_t t = s; t < k; ++t)
				Hessian[s*k+t] += one_minus_rho*Wi[s]*Wi[t];
		}
	}
	for(size_t s = 0; s < k; ++s) {
		for(size_t t = 0; t <= s; ++t)
			Hessian[s*k+t] = Hessian[t*k+s];
		Hessian[s*k+s] += lambda;
	}

	ls_solve_chol(Hessian, y, k);
} // }}}
// Alternating Least Squares for Matrix Factorization with uniform rho
void als_pu(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	if(model.major_type != pmf_model_t::ROWMAJOR) {
		fprintf(stderr, "ALS_PU requires pmf_model_t::ROWMAJOR model\n");
		return;
	}

	int num_threads_old = omp_get_num_threads();
	int maxiter = param.maxiter;
	val_type lambda = param.lambda;
	val_type rho = param.rho;
	omp_set_num_threads(param.threads);
	
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

	// Create transpose view of R
	smat_t &R = training_set, Rt = R.transpose();
	smat_t &testR = test_set, testRt = testR.transpose();
	mat_t &W = model.W, &H = model.H;

	vec_t rhoWTW(model.k*model.k), &rhoHTH = rhoWTW;
	vec_t rhoWTe(model.k), &rhoHTe = rhoWTe;

	std::vector<vec_t> Hessian_set(param.threads, vec_t(model.k*model.k));

	double Wtime = 0, Htime = 0, start = 0;
	for(int iter = 1; iter <= maxiter; ++iter) {
		// Update H
		start = omp_get_wtime();

		compute_rhoWTW(W, rhoWTW, model.k, rho);
#pragma omp parallel for schedule(static)
		for(int t = 0; t < model.k; t++) {
			val_type sum = 0.0;
			for(size_t r = 0; r < R.rows; r++)
				sum += W[r][t];
			rhoWTe[t] = rho*sum;
		}


#pragma omp parallel for schedule(kind) shared(R,H,W)
		for(long c = 0; c < R.cols; ++c) {
			int tid = omp_get_thread_num();
			//val_type local_lambda = lambda*(rho*R.cols+(1-rho)*R.nnz_of_col(c));
			val_type local_lambda = lambda*R.nnz_of_col(c);
			updateOne_pu(H[c], R, c, W, local_lambda, rho, rhoWTW.data(), &Hessian_set[tid][0]);
		}
		Htime += omp_get_wtime() - start;
		// Update W
		start = omp_get_wtime();

		compute_rhoWTW(H, rhoHTH, model.k, rho);
#pragma omp parallel for schedule(static)
		for(int t = 0; t < model.k; t++) {
			val_type sum = 0.0;
			for(size_t c = 0; c < R.cols; c++)
				sum += H[c][t];
			rhoHTe[t] = rho*sum;
		}
#pragma omp parallel for schedule(kind) shared(R,H,W)
		for(long c = 0; c < Rt.cols; ++c) {
			int tid = omp_get_thread_num();
			//val_type local_lambda = lambda*(rho*Rt.cols+(1-rho)*Rt.nnz_of_col(c));
			val_type local_lambda = lambda*Rt.nnz_of_col(c);
			updateOne_pu(W[c], Rt, c, H, local_lambda, rho, rhoHTH.data(), &Hessian_set[tid][0]);
		}
		Wtime += omp_get_wtime() - start;

		if(param.verbose) {
			double loss_omega, loss_zero;
			double loss = compute_pu_loss(R, param, model, &loss_omega, &loss_zero); 
			double reg = compute_reg(R, param, model);

			printf("iter %d time %.10g loss %.10g omega %.10g zero %.10g train-rmse %.10g reg %.10g obj %.10g", 
					iter, Htime+Wtime, loss, loss_omega, loss_zero, sqrt(loss/R.nnz), reg, loss+lambda*reg);
		}

		if(param.do_predict && testR.nnz!=0) {
			double test_loss = compute_loss(testR, param, model);
			printf(" rmse %.10g", sqrt(test_loss/testR.nnz)); 
		}
		if(param.verbose) puts("");
		fflush(stdout);
	}
	omp_set_num_threads(num_threads_old);

} // }}}
