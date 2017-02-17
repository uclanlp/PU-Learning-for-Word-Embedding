#include <cstring>
#include "pmf_matlab.hpp"

static void exit_with_help() {
	mexPrintf(
	"Usage: [val] = pmf_predict(I, J, model)\n"
	"       [val] = pmf_predict(I, J, W, H)\n"
	"       [val] = pmf_predict(I, J, W, H, global_bias=0)\n"
	"     I: row-index array of length=nnz (1-based indices)\n"
	"     J: col-index array of length=nnz (1-based indices)\n"
	"     model: pmf_model\n"
	"     W: m*k dense matrix\n"
	"     H: n*k dense matrix\n"
	"     val is the retured array s.t val(idx)= W(:,I(idx))'*H(:,J(idx))\n"
	);
}

static void fake_answer(mxArray *plhs[]) { plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(nrhs <= 2) {
		exit_with_help();
		return;
	}
	const mxArray *mxI = prhs[0], *mxJ = prhs[1];
	size_t nr_entries = (size_t) mxGetM(mxI);

	mxArray *mxV = plhs[0] = mxCreateDoubleMatrix(nr_entries, 1, mxREAL);
	double *I = mxGetPr(mxI), *J = mxGetPr(mxJ), *pred_val = mxGetPr(mxV);
	const int idx_base = 1;
	if(nrhs == 3) {
		pmf_model_t model = mxStruture_to_pmf_model(prhs[2]);
		model.predict_entries(nr_entries, I, J, pred_val, idx_base);
	} else if(nrhs == 4) {
		pmf_model_t model = gen_pmf_model(prhs[2], prhs[3]);
		model.predict_entries(nr_entries, I, J, pred_val, idx_base);
	} else if(nrhs == 5) {
		pmf_model_t model = gen_pmf_model(prhs[2], prhs[3], *mxGetPr(prhs[4]));
		model.predict_entries(nr_entries, I, J, pred_val, idx_base);
	} else {
		fake_answer(plhs);
		exit_with_help();
		return;
	}
}
