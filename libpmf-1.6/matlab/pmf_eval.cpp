#include <cstring>
#include "pmf_matlab.hpp"

static void exit_with_help() {
	mexPrintf(
	"Usage: [rmse dcg ndcg] = pmf_eval(I, J, true_val, pred_val, topk [, col=0])\n"
	"     I: row-index array of length=nnz (1-based indices)\n"
	"     J: col-index array of length=nnz (1-based indices)\n"
	"     pred_val: pred_val = pmf_predict(I, J, model)\n"
	"Usage: [dcg ndcg] = pmf_eval(trueR, pred_topk [, col=0])\n"
	"     trueR: an m-by-n sparse matrix or an nnz-by-3 dense matrix [I J V], where [I J true_val] = find(trueR)\n"
	"     pred_topk: pred_topk = pmf_predict_topk(I, H, topk, ignored)\n"
	"     col: col-wise evaluation or row-wise evaluation\n"
	);
}

static void fake_answer(mxArray *plhs[]) {
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); 
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL); 
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL); 
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(nrhs != 2 && nrhs !=3 && nrhs != 5 && nrhs != 6) {
		exit_with_help();
		return;
	}

	if(nrhs == 2 || nrhs == 3) {
		const mxArray *mxR = prhs[0], *mxpred = prhs[1];
		bool colwise = nrhs == 2? false : (*mxGetPr(prhs[2]) == 1);
		smat_t trueR;
		mxArray_to_smat(mxR, trueR);
		if(mxGetN(mxpred) != (colwise? trueR.cols : trueR.rows)) {
			mexPrintf("Error: size(trueR, %d) = %ld != size(pred_topk,2) = %ld\n", colwise? 2:1, colwise? trueR.cols:trueR.rows, mxGetN(mxpred));
			fake_answer(plhs);
			return;
		}
		double *pred_topk = mxGetPr(mxpred);
		size_t topk = mxGetM(mxpred);

		const int idx_base = 1;
		double *dcg = mxGetPr(plhs[0] = mxCreateDoubleMatrix(topk,1,mxREAL));
		double *ndcg= mxGetPr(plhs[1] = mxCreateDoubleMatrix(topk,1,mxREAL));

		if(colwise) 
			compute_ndcg_csc_full(trueR.rows, trueR.cols, trueR.col_ptr, trueR.row_idx, trueR.val, pred_topk, topk, dcg, ndcg, idx_base); 
		else 
			compute_ndcg_csr_full(trueR.rows, trueR.cols, trueR.row_ptr, trueR.col_idx, trueR.val_t, pred_topk, topk, dcg, ndcg, idx_base); 

	} else if(nrhs == 5 || nrhs == 6) {
		double *row_idx = mxGetPr(prhs[0]), *col_idx = mxGetPr(prhs[1]);
		double *true_val = mxGetPr(prhs[2]), *pred_val = mxGetPr(prhs[3]);
		int topk = (int)*mxGetPr(prhs[4]); if(topk <= 0) topk = 3;
		bool colwise = nrhs == 5? false : *mxGetPr(prhs[5]) == 1;
		size_t nnz = mxGetM(prhs[0]);
		if(nlhs >= 0) { // RMSE only
			double rmse = 0;
			if(nnz != 0) {
				for(size_t idx = 0; idx < nnz; idx++) {
					double tmp = true_val[idx] - pred_val[idx];
					rmse += tmp*tmp;
				}
				rmse = sqrt(rmse/nnz);
			}
			plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL); *(mxGetPr(plhs[0])) = rmse;
		}
		if(nlhs >= 2) { // NDCG 
			smat_t trueR, predR;
			mxCoo_to_smat(row_idx, col_idx, true_val, nnz, trueR);
			mxCoo_to_smat(row_idx, col_idx, pred_val, nnz, predR);
			double *dcgs = mxGetPr(plhs[1] = mxCreateDoubleMatrix(topk,1,mxREAL));
			double *ndcgs= mxGetPr(plhs[2] = mxCreateDoubleMatrix(topk,1,mxREAL));

			if(colwise) 
				compute_ndcg_csc(trueR.rows, trueR.cols, trueR.col_ptr, trueR.row_idx, trueR.val, predR.val, topk, dcgs, ndcgs);
			else 
				compute_ndcg_csr(trueR.rows, trueR.cols, trueR.row_ptr, trueR.col_idx, trueR.val_t, predR.val_t, topk, dcgs, ndcgs);
		}
		return;
	} else if (nrhs == 3 || nrhs == 4) {
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
