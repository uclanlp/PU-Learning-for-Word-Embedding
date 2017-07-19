#include "pmf.h"
#define kind dynamic,640

// CCD rank-one

inline val_type RankOneUpdate(const smat_t &R, const int j, const vec_t &u, const val_type lambda, const val_type vj, double *redvar, int do_nmf){ // {{{
	val_type g=0, h=lambda;
	if(R.col_ptr[j+1]==R.col_ptr[j]) return 0;
	for(size_t idx=R.col_ptr[j]; idx < R.col_ptr[j+1]; idx++) {
		size_t i = R.row_idx[idx];
		g += R.weight[idx] * u[i]*R.val[idx]; //FIXIT
		h += R.weight[idx] * u[i]*u[i]; //FIXIT
	}
	val_type newvj = g/h, tmp = 0, delta = 0, fundec = 0;
	if((do_nmf>0) & (newvj<0)) {
		newvj = 0;
		delta = vj; // old - new
		fundec = -2*g*vj + h*vj*vj;
	} else {
		delta = vj - newvj;
		fundec = h*delta*delta;
	}
	//val_type delta = vj - newvj;
	//val_type fundec = h*delta*delta;
	//val_type lossdec = fundec - lambda*delta*(vj+newvj);
	//val_type gnorm = (g-h*vj)*(g-h*vj);
	*redvar += fundec;
	//*redvar += lossdec;
	return newvj;
} // }}}

//I use this function a lot of times, need to pay a lot of attention
inline double UpdateRating(smat_t &R, const vec_t &Wt2, const vec_t &Ht2) { // {{{
	double loss=0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(long c = 0; c < R.cols; c++){
			val_type Htc = Ht2[2*c], oldHtc = Ht2[2*c+1];
			double loss_inner = 0;
			for(size_t idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt2[2*R.row_idx[idx]]*Htc-Wt2[2*R.row_idx[idx]+1]*oldHtc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
	return loss;
} // }}}
// To be improved!
static inline double UpdateRating(smat_t &R, const vec_t &Wt, const vec_t &Ht, bool add) { // {{{
	double loss=0;
	if(add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(long c = 0; c < R.cols; c++){
			val_type Htc = Ht[c];
			double loss_inner = 0;
			for(size_t idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] +=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += R.val[idx]*R.val[idx];
				//loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;
	} else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(long c = 0; c < R.cols; c++){
			val_type Htc = Ht[c];
			double loss_inner = 0;
			for(size_t idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += R.val[idx]*R.val[idx];
				//loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;
	}
}// }}}

// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	size_t k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	val_type lambda = param.lambda;
	double eps = param.eps;
	double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	size_t num_updates = 0;
	double reg = 0,loss;

	omp_set_num_threads(param.threads);

	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.W[t][r] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.H[t][c] = 0;
			}
	} // }}}

	// Create transpose view of R
	smat_t &R = training_set, Rt = R.transpose();
	smat_t &testR = test_set, testRt = testR.transpose();
	mat_t &W = model.W, &H = model.H;
	if(param.warm_start) {
		for(size_t t = 0; t < k; t++) {
			loss = UpdateRating(R, W[t], H[t], false);
			loss = UpdateRating(Rt, H[t], W[t], false);
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
			for(size_t c = 0; c < R.cols; c++)
				reg += H[t][c]*H[t][c]*R.nnz_of_col(c);
			oldobj = loss + param.lambda * reg;
			if(param.verbose)
				printf("iter 0 rank %ld loss %.10g obj %.10g reg %.7g ", t, loss, oldobj, reg);

			if(param.do_predict && testR.nnz!= 0) {
				double test_loss;
				test_loss = UpdateRating(testR, W[t], H[t], false);
				test_loss = UpdateRating(testRt, H[t], W[t], false);
				printf("rmse %.10g\n", sqrt(test_loss/testR.nnz));
			}
		}
	} else {
		// initial value of the regularization term
		// H is a zero matrix now.
		for(size_t t = 0;t < k; t++)
			for(size_t c = 0; c < R.cols; c++)
				if(R.nnz_of_col(c) > 0)
					H[t][c] = 0;
		for(size_t t = 0;t < k; t++)
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
	}

	for(int oiter = 1; oiter <= maxiter; ++oiter) {
		double gnorm = 0, initgnorm=0;
		double rankfundec = 0;
		double fundec_max = 0;
		int early_stop = 0;
		for(size_t tt=0; tt < k; tt++) {
			size_t t = tt;
			if(early_stop >= 5) break;
			//if(oiter>1) { t = rand()%k; }
			start = omp_get_wtime();
			vec_t &u = W[t], &v= H[t];

			// Create Rhat = R + Wt Ht^T
			if (param.warm_start || oiter > 1) {
				UpdateRating(R, u, v, true);
				UpdateRating(Rt, v, u, true);
			}
			Itime += omp_get_wtime() - start;

			if (param.warm_start || oiter > 1) {
				if(param.do_predict && testR.nnz!=0) {
					UpdateRating(testR, u, v, true);
					UpdateRating(testRt, v, u, true);
				}
			}
			for(size_t c = 0; c < R.cols; c++) reg -= v[c]*v[c]*R.nnz_of_col(c);
			for(size_t r = 0; r < R.rows; r++) reg -= u[r]*u[r]*R.nnz_of_row(r);

			gnorm = 0, initgnorm = 0;
			double innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter;
			//	if(oiter > 1) maxit *= 2;
			for(int iter = 1; iter <= maxit; ++iter){
				
                                
                                
                                if (t != param.k - 1){
                                // Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < R.cols; ++c) {
					size_t nz = R.nnz_of_col(c);
					if(nz)
						v[c] = RankOneUpdate(R, c, u, lambda*nz, v[c], &innerfundec_cur, param.do_nmf);
				}
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;
                                }
				
                                
                                
                                if (t != param.k - 2){
				// Update W[t]
				start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < Rt.cols; ++c) {
					size_t nz = Rt.nnz_of_col(c);
					if(nz)
						u[c] = RankOneUpdate(Rt, c, v, lambda*nz, u[c], &innerfundec_cur, param.do_nmf);
				}
				num_updates += Rt.cols;
                                }





				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break;
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = std::max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = std::max(fundec_max, innerfundec_cur);
				Wtime += omp_get_wtime() - start;
			}

			// Update R and Rt
			start = omp_get_wtime();
			loss = UpdateRating(R, u, v, false);
			loss = UpdateRating(Rt, v, u, false);
			Rtime += omp_get_wtime() - start;

			for(size_t c = 0; c < R.cols; c++) reg += v[c]*v[c]*R.nnz_of_col(c);
			for(size_t r = 0; r < R.rows; r++) reg += u[r]*u[r]*R.nnz_of_row(r);

			double obj = loss+reg*lambda;
			if(param.verbose)
				printf("iter %d rank %lu time %.10g loss %.10g obj %.10g diff %.10g gnorm %.6g reg %.7g ",
						oiter, t+1, Htime+Wtime+Rtime, loss, obj, oldobj - obj, initgnorm, reg);
			oldobj = obj;

			if(param.do_predict && testR.nnz!=0) {
				double test_loss = 0;
				test_loss = UpdateRating(testR, u, v, false);
				test_loss = UpdateRating(testRt, v, u, false);
				printf("rmse %.10g", sqrt(test_loss/testR.nnz));
			}
			if(param.verbose) puts("");
			fflush(stdout);
		}
	}
	omp_set_num_threads(num_threads_old);
} // }}}

// Cyclic Coordinate Descent for Matrix Factorization Speedup Version
void ccdr1_speedup(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model){ // {{{
	long k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	val_type lambda = param.lambda;
	double eps = param.eps;
	double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	size_t num_updates = 0;
	double reg = 0,loss;

	omp_set_num_threads(param.threads);

	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.W[t][r] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.H[t][c] = 0;
			}
	} // }}}

	// Create transpose view of R
	smat_t &R = training_set, Rt = R.transpose();
	smat_t &testR = test_set, testRt = testR.transpose();
	mat_t &W = model.W, &H = model.H;

	if(param.warm_start) {
		for(size_t t = 0; t < k; t++) {
			loss = UpdateRating(R, W[t], H[t], false);
			loss = UpdateRating(Rt, H[t], W[t], false);
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
			for(size_t c = 0; c < R.cols; c++)
				reg += H[t][c]*H[t][c]*R.nnz_of_col(c);
			oldobj = loss + param.lambda * reg;
			if(param.verbose)
				printf("iter 0 rank %lu loss %.10g obj %.10g reg %.7g ", t, loss, oldobj, reg);

			if(param.do_predict && testR.nnz!= 0) {
				double test_loss;
				test_loss = UpdateRating(testR, W[t], H[t], false);
				test_loss = UpdateRating(testRt, H[t], W[t], false);
				printf("rmse %.10g\n", sqrt(test_loss/testR.nnz));
			}
		}
	} else {
		// initial value of the regularization term
		// H is a zero matrix now.
		for(size_t t = 0;t < k; t++)
			for(size_t c = 0; c < R.cols; c++)
				if(R.nnz_of_col(c) > 0)
					H[t][c] = 0;
		for(size_t t = 0;t < k; t++)
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
	}

	size_t adaptive_k = 1;  // upper_bound = k
	val_type adaptive_lambda = lambda*log2((double)k); //*k/adaptive_k; // lower_bound = lambda

	for(int oiter = 1; oiter <= maxiter; ++oiter) {
		double gnorm = 0, initgnorm=0;
		double rankfundec = 0;
		double fundec_max = 0;
		int early_stop = 0;
		size_t tt = 0;
		//for(size_t tt=0; tt < k; tt++) {
		while(tt < adaptive_k) {
			size_t t = tt; tt++;
			if(early_stop >= 5) break;
			//if(oiter>1) { t = rand()%k; }
			start = omp_get_wtime();
			vec_t &u = W[t], &v= H[t];

			// Create Rhat = R + Wt Ht^T
			if (param.warm_start || oiter > 1) {
				UpdateRating(R, u, v, true);
				UpdateRating(Rt, v, u, true);
			}
			Itime += omp_get_wtime() - start;

			if (param.warm_start || oiter > 1) {
				if(param.do_predict && testR.nnz!=0) {
					UpdateRating(testR, u, v, true);
					UpdateRating(testRt, v, u, true);
				}
			}
			for(size_t c = 0; c < R.cols; c++) reg -= v[c]*v[c]*R.nnz_of_col(c);
			for(size_t r = 0; r < R.rows; r++) reg -= u[r]*u[r]*R.nnz_of_row(r);

			gnorm = 0, initgnorm = 0;
			double innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter;
			//	if(oiter > 1) maxit *= 2;
			for(int iter = 1; iter <= maxit; ++iter){
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < R.cols; ++c) {
					size_t nz = R.nnz_of_col(c);
					if(nz)
						v[c] = RankOneUpdate(R, c, u, adaptive_lambda*nz, v[c], &innerfundec_cur, param.do_nmf);
				}
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;
				// Update W[t]
				start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < Rt.cols; ++c) {
					size_t nz = Rt.nnz_of_col(c);
					if(nz)
						u[c] = RankOneUpdate(Rt, c, v, adaptive_lambda*nz, u[c], &innerfundec_cur, param.do_nmf);
				}
				num_updates += Rt.cols;
				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break;
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = std::max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = std::max(fundec_max, innerfundec_cur);
				Wtime += omp_get_wtime() - start;
			}

			// Update R and Rt
			start = omp_get_wtime();
			loss = UpdateRating(R, u, v, false);
			loss = UpdateRating(Rt, v, u, false);
			Rtime += omp_get_wtime() - start;

			for(size_t c = 0; c < R.cols; c++) reg += v[c]*v[c]*R.nnz_of_col(c);
			for(size_t r = 0; r < R.rows; r++) reg += u[r]*u[r]*R.nnz_of_row(r);

			double obj = loss+reg*lambda;
			if(param.verbose)
				printf("iter %d rank %lu time %.10g loss %.10g train-rmse %.10g obj %.10g diff %.10g gnorm %.6g reg %.7g ",
						oiter, t+1, Htime+Wtime+Rtime, loss, sqrt(loss/R.nnz), obj, oldobj - obj, initgnorm, reg);
			oldobj = obj;

			if(param.do_predict && testR.nnz!=0) {
				double test_loss = 0;
				test_loss = UpdateRating(testR, u, v, false);
				test_loss = UpdateRating(testRt, v, u, false);
				printf("rmse %.10g", sqrt(test_loss/testR.nnz));
			}
			if(param.verbose) puts("");
			fflush(stdout);
		}

		if(tt == adaptive_k) {
			adaptive_k = std::min(2*adaptive_k, (size_t)k);
			//adaptive_lambda = std::max(lambda*k/adaptive_k, lambda);
			adaptive_lambda = std::max(adaptive_lambda-lambda, lambda);
			if(param.verbose>1)
				printf("---------> adaptive (k,lambda) = (%ld, %g)\n", adaptive_k, adaptive_lambda);
		}

	}
	omp_set_num_threads(num_threads_old);
} // }}}







// CCD++PU

static inline val_type dot(const vec_t& u, const vec_t& v) { // {{{
	val_type ret = 0.0;
	const size_t k = u.size();
#pragma omp parallel for schedule(static) reduction(+:ret) shared(u) 
	for(int t = 0; t < k; t++)
		ret += u[t]*v[t];
	return ret;
} // }}}
//I also use this function, but it will not affects the result.
static double compute_pu_loss(smat_t &A, smat_t &R, pmf_parameter_t &param, pmf_model_t &model, double *loss_omega=NULL, double *loss_zero=NULL) { // {{{
	double omega_part = 0.0;
#pragma omp parallel for schedule(static) reduction(+:omega_part)
	for(long idx = 0; idx < R.nnz; idx++) {
		val_type tmp = R.val[idx];
		//loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
		omega_part += tmp*tmp;
	}

	double zero_part = 0.0;
#pragma omp parallel for schedule(static) reduction(+:zero_part)
	for(long idx = 0; idx < R.nnz; idx++) {
		val_type tmp = R.val[idx]-A.val[idx];
		zero_part -= tmp*tmp;
	}

	bool transpose = true;
	vec_t HTH(model.k*model.k), WTW(model.k*model.k);
	val_type *W = model.W.data(), *H = model.H.data();
	dmat_x_dmat_colmajor(val_type(1.0), W, transpose, W, !transpose, val_type(0.0), WTW.data(), model.k, model.k, model.rows);
	dmat_x_dmat_colmajor(val_type(1.0), H, transpose, H, !transpose, val_type(0.0), HTH.data(), model.k, model.k, model.cols);
	zero_part += do_dot_product(WTW.data(), HTH.data(), model.k*model.k);
	zero_part *= param.rho;

	if(loss_omega) *loss_omega = omega_part;
	if(loss_zero) *loss_zero = zero_part;

	return zero_part+omega_part;
} // }}}

//Here Hsiang-fu make modifications, pay special attention!
void pu_rank_one_update(int cur_t, smat_t &A, smat_t &R, pmf_parameter_t &param, mat_t &W, mat_t &H, vec_t &u, vec_t &v, vec_t &uTWHT, double &innerfundec_cur) {  // {{{
        
	// could be replaced by BLAS operations
	val_type uTu = dot(u, u); 
#pragma omp parallel for schedule(kind) shared(uTWHT) 
	for(long c=0; c < A.cols; c++)
		uTWHT[c] = 0;

	for(int t=0; t<param.k; ++t) {
		if(t == cur_t) continue;
		val_type uTWt = dot(u, W[t]);
		vec_t &Ht = H[t];
#pragma omp parallel for schedule(kind) shared(v,W,H) 
		for(int c=0; c<R.cols; c++) 
			uTWHT[c] += uTWt * Ht[c];
	} 

	val_type fun_dec = 0;
#pragma omp parallel for schedule(kind) shared(R,H,W) reduction(+:fun_dec)
	for(long c = 0; c < R.cols; c++) {
		if(R.nnz_of_col(c) ==0 && param.rho == 0) { continue ;}

		val_type uRc = 0, uTu_omegac = 0, uAc = 0;

                //##########################
                if (param.glove_weight){//boundary! if I use glove weight

                       // printf("in pu_rank_one_update, now have weight\n");
        	for(size_t idx = R.col_ptr[c] ; idx<R.col_ptr[c+1] ; idx++ ) {
			size_t r = R.row_idx[idx];
			uRc += (R.weight[idx]-param.rho) * u[r]*R.val[idx];//FIXIT
			uAc += u[r]*A.val[idx];//FIXIT
                        uTu_omegac += (R.weight[idx]-param.rho) * u[r]*u[r];//FIXIT
		}

		double neg_fp = (uRc)+param.rho*(uAc-uTWHT[c]);  //FIXME
		double fpp = (uTu_omegac + param.rho*uTu + param.lambda*R.nnz_of_col(c));//FIXME
		double vc_new = neg_fp/fpp;
		fun_dec += fpp*(vc_new-v[c])*(vc_new-v[c]);
		v[c] = vc_new;
                
                }else//boundary, if I don't use glove weight
                
                {
                        //printf("in pu_rank_one_update, now don't have weight\n");
                //###########################
		for(size_t idx = R.col_ptr[c] ; idx<R.col_ptr[c+1] ; idx++ ) {
			size_t r = R.row_idx[idx];
			uRc += u[r]*R.val[idx];
			uAc += u[r]*A.val[idx];
			uTu_omegac += u[r]*u[r];
		}

		double neg_fp = (1-param.rho)*(uRc)+param.rho*(uAc-uTWHT[c]);  
		double fpp = ((1-param.rho)*uTu_omegac + param.rho*uTu + param.lambda*R.nnz_of_col(c));
		double vc_new = neg_fp/fpp;
		fun_dec += fpp*(vc_new-v[c])*(vc_new-v[c]);
		v[c] = vc_new;

                }
              //###########################
	}
	innerfundec_cur += fun_dec;
} // }}}

// Cyclic Coordinate Descent for Matrix Factorization with uniform rho

//this is the main functionn for ccdr1_pu, look it carefully


void ccdr1_pu(smat_t &training_set, smat_t &test_set, pmf_parameter_t &param, pmf_model_t &model, int do_shuffle, std::vector<unsigned> row_perm, std::vector<unsigned> col_perm, std::vector<unsigned> inverse_row_perm, std::vector<unsigned> inverse_col_perm, const char *model_file_name){ // {{{
	size_t k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	val_type lambda = param.lambda;
	double eps = param.eps;
	double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	size_t num_updates = 0;
	double reg = 0,loss;

	omp_set_num_threads(param.threads);

        //I don't use this part****************************************
	if(param.remove_bias) { // {{{
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
		for(size_t r = 0; r < training_set.rows; r++)
			if(training_set.nnz_of_row(r) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.W[t][r] = 0;
			}
		for(size_t c = 0; c < training_set.cols; c++)
			if(training_set.nnz_of_col(c) == 0) {
				for(size_t t = 0; t < model.k; t++)
					model.H[t][c] = 0;
			}
	} // }}}
        //I don't use this part****************************************
	
        //copy  &A, At, &testR, testRt
        // Create transpose view of A and R
	smat_t &A = training_set, At = A.transpose();
	smat_t &testR = test_set, testRt = testR.transpose();
	smat_iterator_t<val_type> it(A);//here I announce a iterator, the return of next() is a tuples contains (x, y, val)

        //here, announce R, R just copy A's information, and I modified the code to copy weight information from A to R
 
	smat_t R; R.load_from_iterator(A.rows, A.cols, A.nnz, &it); 
        
	if (param.glove_weight){ 
        for(size_t idx=0; idx < A.nnz; idx++){
        R.weight[idx] = A.weight[idx];//(val[idx]>x_max)?1:pow(val[idx]/x_max, alpha);
        R.weight_t[idx] = A.weight_t[idx];//(val_t[idx]>x_max)?1:pow(val_t[idx]/x_max, alpha);
        }  
	}
        
        smat_t Rt = R.transpose();

        //here I announce two dense matrix, from the deminsion aspect:
        //A = W* H  
        //m*n = m*k * (n*k)T
	
        mat_t &W = model.W, &H = model.H;
        //这里实际上是一个dense vector，在这里使用长度来初始化的，uu是一个竖条，vv是一个横行?这里我是不是理解错了
	vec_t uu(R.rows), vv(R.cols);

	if(param.warm_start) { // {{{
		for(size_t t = 0; t < k; t++) {
			UpdateRating(R, W[t], H[t], false);
			UpdateRating(Rt, H[t], W[t], false);
			double loss_omega, loss_zero;
			loss = compute_pu_loss(A, R, param, model, &loss_omega, &loss_zero); 
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
			for(size_t c = 0; c < R.cols; c++)
				reg += H[t][c]*H[t][c]*R.nnz_of_col(c);
			oldobj = loss + param.lambda * reg;
			if(param.verbose)
				printf("iter 0 rank %ld loss %.10g omega %.10g zero %.10g obj %.10g reg %.7g ", t, loss, loss_omega, loss_zero, oldobj, reg);

			if(param.do_predict && testR.nnz!= 0) {
				double test_loss;
				test_loss = UpdateRating(testR, W[t], H[t], false);
				test_loss = UpdateRating(testRt, H[t], W[t], false);
				printf("rmse %.10g\n", sqrt(test_loss/testR.nnz));
			}
		}
	} else {
		// initial value of the regularization term
		// H is a zero matrix now.
		for(size_t t = 0;t < k; t++)//k就是k
			for(size_t c = 0; c < R.cols; c++)//R.cols是n
				if(R.nnz_of_col(c) > 0)
					H[t][c] = 0;//这里处理的矩阵的维度是k*n，先是每一行循环，然后每一列循环，只要这一列有一个0，这个点就置0
		for(size_t t = 0;t < k; t++)
			for(size_t r = 0; r < R.rows; r++)
				reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
	} // }}}//what does reg mean here
        //H is k*n, W is k*r?
        if (param.glove_bias){
        for (int idx = 0; idx < model.W[0].size(); idx++){
                model.W[param.k - 2][idx] = 1;
                model.W[param.k - 1][idx] = 0;
                model.H[param.k - 1][idx] = 1;
        }
        }

	for(int oiter = 1; oiter <= maxiter; ++oiter) {//this is outer iteration
		double gnorm = 0, initgnorm=0;
		double rankfundec = 0;//what does this abbreviation mean?
		double fundec_max = 0;//what does this abbreviation mean?
		int early_stop = 0;
		for(size_t tt=0; tt < k; tt++) {//here k is rank, do iteration by each rank
			size_t t = tt;
			if(early_stop >= 5) break;
			//if(oiter>1) { t = rand()%k; }
			start = omp_get_wtime();
			vec_t &u = W[t], &v= H[t];//please, there update by feature
                        //this part deals with formula 3.16
			// Create Rhat = R + Wt Ht^T
			if (param.warm_start || oiter > 1) {//  || is or
				UpdateRating(R, u, v, true);
				UpdateRating(Rt, v, u, true);
			}
			Itime += omp_get_wtime() - start;
                        
                        //I don't need this part, bucause I don't do predict###############
			if (param.warm_start || oiter > 1) {
				if(param.do_predict && testR.nnz!=0) {
					UpdateRating(testR, u, v, true);
					UpdateRating(testRt, v, u, true);
				}
			}
                        //I don't need this part, bucause I don't do predict###############
                        if(param.verbose) {
				for(size_t c = 0; c < R.cols; c++) reg -= v[c]*v[c]*R.nnz_of_col(c);
				for(size_t r = 0; r < R.rows; r++) reg -= u[r]*u[r]*R.nnz_of_row(r);
			}

                        //pu_rank_one_update if formula 3.10
			gnorm = 0, initgnorm = 0;
			double innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter;
			//	if(oiter > 1) maxit *= 2;
			for(int iter = 1; iter <= maxit; ++iter){//there decisec how many inner iterations
                            if(param.glove_bias){
                                if (t != param.k -1){
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
				pu_rank_one_update(t, A, R, param, W, H, u, v, vv, innerfundec_cur);
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;
                                }

                                if (t != param.k -2){
				// Update W[t]
				start = omp_get_wtime();
				pu_rank_one_update(t, At, Rt, param, H, W, v, u, uu, innerfundec_cur);//
				num_updates += Rt.cols;
				Wtime += omp_get_wtime() - start;
                                }
                            }
                            else{
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
				pu_rank_one_update(t, A, R, param, W, H, u, v, vv, innerfundec_cur);
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;

				// Update W[t]
				start = omp_get_wtime();
				pu_rank_one_update(t, At, Rt, param, H, W, v, u, uu, innerfundec_cur);//why update W[t] needs At and Rt?
				num_updates += Rt.cols;
				Wtime += omp_get_wtime() - start;

                            }

				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break;
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = std::max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = std::max(fundec_max, innerfundec_cur);
			}//the end of inner iteration
                        
			// Update R and Rt
			start = omp_get_wtime();
			UpdateRating(R, u, v, false);
			UpdateRating(Rt, v, u, false);
			Rtime += omp_get_wtime() - start;

			if(param.verbose) {
				double loss_omega, loss_zero;
				loss = compute_pu_loss(A, R, param, model, &loss_omega, &loss_zero); 
				for(size_t c = 0; c < R.cols; c++) reg += v[c]*v[c]*R.nnz_of_col(c);
				for(size_t r = 0; r < R.rows; r++) reg += u[r]*u[r]*R.nnz_of_row(r);

				double obj = loss+reg*lambda;
				printf("iter %d rank %lu time %.10g loss %.10g omega %.10g zero %.10g obj %.10g diff %.10g gnorm %.6g reg %.7g ",
						oiter, t+1, Htime+Wtime+Rtime, loss, loss_omega, loss_zero, obj, oldobj - obj, initgnorm, reg);
				oldobj = obj;
			}
                        //I don't need this part################
			if(param.do_predict && testR.nnz!=0) {
				double test_loss = 0;
				test_loss = UpdateRating(testR, u, v, false);
				test_loss = UpdateRating(testRt, v, u, false);
				printf("rmse %.10g", sqrt(test_loss/testR.nnz));
			}
                        //I don't need this part################
			if(param.verbose) {
				puts("");
				fflush(stdout);
			}
		}//the end of updating each rank


	        if(param.save_each){
                        FILE *model_fpw = NULL;	
                        FILE *model_fph = NULL;//FIXIT

                        float rho = param.rho;
                        float lambda = param.lambda;
                        
                        if(model_file_name) {
                                char matrixname[1024];
                                sprintf(matrixname, "%s-l%f-r%f-oiter%d-gweight%d-xmax%d-gbias%d.W", model_file_name, lambda, rho, oiter, param.glove_weight, param.x_max, param.glove_bias);
                                model_fpw = fopen(matrixname, "w");
                                if(model_fpw == NULL) {
                                        fprintf(stderr,"Error: can't open model file %s\n", model_file_name);
                                        exit(1);
                                }	
                               // sprintf(matrixname, "%s-l%f-r%f-oiter%d.H", model_file_name, lambda, rho, oiter);
                                sprintf(matrixname, "%s-l%f-r%f-oiter%d-gweight%d-xmax%d-gbias%d.H", model_file_name, lambda, rho, oiter, param.glove_weight, param.x_max, param.glove_bias);
                                model_fph = fopen(matrixname, "w");
                                if(model_fph == NULL) {
                                        fprintf(stderr,"Error: can't open model file %s\n", model_file_name);
                                        exit(1);
                                }
                        }
                        if(model_fpw) {
                                if(do_shuffle)
                                        model.apply_permutation(row_perm, col_perm);
                                model.save_embedding(model_fpw,model_fph, param.glove_bias);//FIXIT
                                fclose(model_fpw);
                                fclose(model_fph);
                                if(do_shuffle)
                                        model.apply_permutation(inverse_row_perm, inverse_col_perm);
                        }
                }//end of save_each partition
	}//end of outer iteration
	omp_set_num_threads(num_threads_old);
} // }}}

