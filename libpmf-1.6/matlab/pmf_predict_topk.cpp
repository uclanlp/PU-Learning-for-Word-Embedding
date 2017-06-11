#include <cstring>
#include "pmf_matlab.hpp"

static void exit_with_help() {
	mexPrintf(
	"Usage: [pred_topk] = pmf_predict_topk(I, W, H, topk, ignored)\n"
	"     I: row-index array (1-based indices)\n"
	"     W: m*k dense matrix\n"
	"     H: n*k dense matrix\n"
	"     ignored: a sparse matrix or COO full matrix, use [] to denote no ignore\n"
	"     pred_topk is topk*length(row_idx) array with the topk items in each column\n"
	" Use [pred_topk] = pmf_predict_topk(I, H, W, topk, transpose(ignored)) for column prediction\n"
	"   transpose(ignored) = ignored' if ignored is a sparse matrix\n"
	"                      = [ignored(:,2) ignored(:,1)] if ignored is COO\n"
	);
}

static void fake_answer(mxArray *plhs[]) { plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); }


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(nrhs != 5) {
		exit_with_help();
		fake_answer(plhs);
		return;
	}
	const mxArray *mxI = prhs[0], *mxW = prhs[1], *mxH = prhs[2], *mxTopK = prhs[3], *mxIG = prhs[4];
	if(mxGetN(mxW) != mxGetN(mxH)) {
		mexPrintf("Error: size(W,2) != size(H,2)\n");
		fake_answer(plhs);
		return;
	}

	size_t topk = (size_t) *mxGetPr(mxTopK), pred_rows = (size_t) (mxGetM(mxI)*mxGetN(mxI));
	size_t rows = (size_t) mxGetM(mxW), cols = (size_t) mxGetM(mxH), dim = (size_t) mxGetN(mxW);

	double *I = mxGetPr(mxI); 

	const bool model_init = false;
	pmf_model_t model(rows, cols, dim, pmf_model_t::ROWMAJOR, model_init);
	mat_t &W = model.W, &H = model.H;

	mxDense_to_matRow(mxW, W);
	mxDense_to_matRow(mxH, H);

	smat_t ignored;
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(mxIG, ignored, rows, cols);
	const int idx_base = 1;
	double *pred_topk = mxGetPr(plhs[0]=mxCreateDoubleMatrix(topk, pred_rows, mxREAL));

	int nr_threads = omp_get_num_procs();
	//printf("nr_threads %d\n", nr_threads);
	omp_set_num_threads(nr_threads);
	std::vector<info_t> info_set(nr_threads); 

	for(int th = 0; th < nr_threads; th++) {
		info_set[th].sorted_idx.reserve(cols);
		info_set[th].true_rel.reserve(cols);
		info_set[th].pred_val.reserve(cols);
	}

#pragma omp parallel for
	for(long i = 0; i < pred_rows; i++) {
		info_t &info = info_set[omp_get_thread_num()];
		info.sorted_idx.resize(cols);
		info.pred_val.resize(cols);

		size_t row = (size_t) I[i] - idx_base;

		size_t nr_ignore = 0;
		unsigned *ignore_list  = NULL;
		if(ignored.nnz > 0) {
			nr_ignore = ignored.nnz_of_row(row);
			ignore_list = &ignored.col_idx[ignored.row_ptr[row]];
		}

		size_t valid_len = 0; // assigned by pmf_prepare_candidates
		pmf_prepare_candidates(cols, info.pred_val.data(), info.sorted_idx.data(), valid_len, nr_ignore, ignore_list);
		model.predict_row(row, valid_len, info.sorted_idx.data(), info.pred_val.data());
		sort_idx_by_val(info.pred_val.data(), valid_len, info.sorted_idx.data(), topk);
		for(int t = 0; t < topk; t++) 
			pred_topk[topk*i + t] = t<cols? (double) (info.sorted_idx[t] + idx_base): 0.0;
	}
}
