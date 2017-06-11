#include <cstring>

#include "pmf_matlab.hpp"

bool do_shuffle = true;

static void exit_with_help() { // {{{
	mexPrintf(
	"Usage: [model walltime] = pmf_train(R, testR, W, H [, 'pmf_options'])\n"
	"       [model walltime] = pmf_train(R, testR, [, 'pmf_options'])\n"
	"     R: an m-by-n sparse matrix or an nnz-by-3 dense matrix [I J V], where [I J V] = find(R)\n"
	"     testR: the same format as R, use [] to denote no testing during training\n"
	"     W: initial m-by-k dense double matrix\n"
	"     H: initial n-by-k dense double matrix\n"
	"        If intial W and H are given, \"rank\" will equal to size(W,2).\n"
	"     model: is a structure with three fields (W, H, global_bias):\n"
	"options:\n"
	"    -s type : set type of solver (default 0)\n"
	"    	 0 -- CCDR1 with fundec stopping condition\n"
	"    	 1 -- ALS\n"
	"    	 2 -- SGD\n"
	"    	 9 -- CCDR1 with adaptive ranking increasing\n"
	"    	10 -- PU-CCDR1\n"
	"    	11 -- PU-ALS\n"
	"    	12 -- PU-SGD\n"
	"    	22 -- PU-SGD-ORIG\n"
	"    -k rank : set the rank (default 10)\n"
	"    -n threads : set the number of threads (default 4)\n"
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"
	"    -r rho : set the parameter rho for PU formulation (default 0.01)\n"
	"    -P pu_type: set the types of SGD for PU formulation (default 0)\n"
	"        0 -- PU0:\n"
	"        1 -- PU1:\n"
	"    -t max_iter: set the number of iterations (default 5)\n"
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n"
	"    -S shuffle: random shuffle for rows and columns (default 1)\n"
	"    -b remove_bias: remove bias or not (default 1)\n"
	//"    -p do_predict: do prediction or not (default 0)\n"
	);
} // }}}

// nrhs == 2 or 3 => pmf_train(R, testR [, 'pmf_options']);
// nrhs == 4 or 5 => pmf_train(R, testR, W, H [, 'pmf_options']);
pmf_parameter_t parse_command_line(int nrhs, const mxArray *prhs[]) { // {{{
	pmf_parameter_t param;   // default values have been set by the constructor
	int i, argc = 1;
	int option_pos = -1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	if(nrhs < 1)
		return param;

	// put options in argv[]
	if(nrhs == 3) option_pos = 2;
	if(nrhs == 5) option_pos = 4;
	if(option_pos>0)
	{
		mxGetString(prhs[option_pos], cmd,  mxGetN(prhs[option_pos]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 'r':
				param.rho = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				param.eta0 = atof(argv[i]);
				break;

			case 'B':
				param.nr_blocks = atoi(argv[i]);
				break;

			case 'm':
				param.lrate_method = atoi(argv[i]);
				break;

			case 'u':
				param.betaup = atof(argv[i]);
				break;

			case 'd':
				param.betadown = atof(argv[i]);
				break;

			case 'a':
				param.alpha = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'P':
				param.pu_type = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			case 'S':
				do_shuffle = atoi(argv[i]);
				break;

			case 'b':
				param.remove_bias = atof(argv[i]);
				break;

			default:
				mexPrintf("unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict != 0 && param.verbose == 0)
		param.verbose = 1;

	if (nrhs >= 4) {
		if(mxGetN(prhs[2]) != mxGetN(prhs[3]))
			mexPrintf("Dimensions of W and H do not match!\n");
		int k = (int)mxGetN(prhs[2]);
		if(k != param.k) {
			param.k = k;
			mexPrintf("Change param.k to %d.\n", param.k);
		}
	}

	return param;
} // }}}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}



int run_ccdr1(mxArray *plhs[], int nrhs, const mxArray *prhs[], pmf_parameter_t &param) { // {{{
	mxArray *mxW, *mxH;
	smat_t training_set, test_set;

	size_t tmp_rows = nrhs>3? mxGetM(prhs[2]): 0, tmp_cols = nrhs>3? mxGetM(prhs[3]): 0;
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(prhs[0], training_set, tmp_rows, tmp_cols);
	mxArray_to_smat(prhs[1], test_set, training_set.rows, training_set.cols);

	// fix random seed to have same results for each run
	// (for random initialization)
	long seed = 0L;
	pmf_model_t model(training_set.rows, training_set.cols, param.k, pmf_model_t::COLMAJOR);
	mat_t& W = model.W, &H = model.H;

	// Initialization of W and H
	if(nrhs >= 4) {
		mxDense_to_matCol(prhs[2], W);
		mxDense_to_matCol(prhs[3], H);
	} else {
		model.rand_init(seed);
	}

	if(param.remove_bias) {
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
	}

	// Random permutation for rows and cols of training_set for better load balancing
	std::vector<unsigned> row_perm, inverse_row_perm;
	std::vector<unsigned> col_perm, inverse_col_perm;
	if(do_shuffle) {
		gen_permutation_pair(training_set.rows, row_perm, inverse_row_perm);
		gen_permutation_pair(training_set.cols, col_perm, inverse_col_perm);

		training_set.apply_permutation(row_perm, col_perm);
		test_set.apply_permutation(row_perm, col_perm);
		if(nrhs >= 4)
			model.apply_permutation(inverse_row_perm, inverse_col_perm);
	}

	// Execute the program
	double time = omp_get_wtime();
	if(param.solver_type == CCDR1)
		ccdr1(training_set, test_set, param, model);
	else if(param.solver_type == CCDR1_SPEEDUP)
		ccdr1_speedup(training_set, test_set, param, model);
	else if(param.solver_type == PU_CCDR1)
		ccdr1_pu(training_set, test_set, param, model);
	double walltime = omp_get_wtime() - time;

	if(do_shuffle) // recover the permutation for the model
		model.apply_permutation(row_perm, col_perm);

	// Write back the result
	plhs[0] = pmf_model_to_mxStruture(model);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*(mxGetPr(plhs[1])) = walltime;
	return 0;
} // }}}

int run_als(mxArray *plhs[], int nrhs, const mxArray *prhs[], pmf_parameter_t &param) { // {{{
	mxArray *mxW, *mxH;
	smat_t training_set, test_set;

	size_t tmp_rows = nrhs>3? mxGetM(prhs[2]): 0;
	size_t tmp_cols = nrhs>3? mxGetM(prhs[3]): 0;
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(prhs[0], training_set, tmp_rows, tmp_cols);
	mxArray_to_smat(prhs[1], test_set, training_set.rows, training_set.cols);

	// fix random seed to have same results for each run
	// (for random initialization)
	long seed = 0L;
	// ALS requires rowmajor model
	pmf_model_t model(training_set.rows, training_set.cols, param.k, pmf_model_t::ROWMAJOR);
	mat_t& W = model.W, &H = model.H;

	// Initialization of W and H
	if(nrhs >= 4) {
		mxDense_to_matRow(prhs[2], W);
		mxDense_to_matRow(prhs[3], H);
	} else {
		model.rand_init(seed);
	}

	if(param.remove_bias) {
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
	}

	// Random permutation for rows and cols of training_set for better load balancing
	std::vector<unsigned> row_perm, inverse_row_perm;
	std::vector<unsigned> col_perm, inverse_col_perm;
	if(do_shuffle) {
		gen_permutation_pair(training_set.rows, row_perm, inverse_row_perm);
		gen_permutation_pair(training_set.cols, col_perm, inverse_col_perm);

		training_set.apply_permutation(row_perm, col_perm);
		test_set.apply_permutation(row_perm, col_perm);
		if(nrhs >= 4)
			model.apply_permutation(inverse_row_perm, inverse_col_perm);
	}

	// Execute the program
	double time = omp_get_wtime();
	if(param.solver_type == ALS)
		als(training_set, test_set, param, model);
	else if(param.solver_type == PU_ALS)
		als_pu(training_set, test_set, param, model);
	double walltime = omp_get_wtime() - time;

	if(do_shuffle) // recover the permutation for the model
		model.apply_permutation(row_perm, col_perm);

	// Write back the result
	plhs[0] = pmf_model_to_mxStruture(model);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*(mxGetPr(plhs[1])) = walltime;

	return 0;
} // }}}

int run_sgd(mxArray *plhs[], int nrhs, const mxArray *prhs[], pmf_parameter_t &param) { // {{{
	mxArray *mxW, *mxH;
	blocks_t training_set, test_set;

	size_t tmp_rows = nrhs>3? mxGetM(prhs[2]): 0;
	size_t tmp_cols = nrhs>3? mxGetM(prhs[3]): 0;
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(prhs[0], training_set, tmp_rows, tmp_cols);
	mxArray_to_smat(prhs[1], test_set, training_set.rows, training_set.cols);

	// fix random seed to have same results for each run
	// (for random initialization)
	long seed = 0L;
	// SGD requires rowmajor model
	pmf_model_t model(training_set.rows, training_set.cols, param.k, pmf_model_t::ROWMAJOR);
	mat_t& W = model.W, &H = model.H;

	// Initialization of W and H
	if(nrhs >= 4) {
		mxDense_to_matRow(prhs[2], W);
		mxDense_to_matRow(prhs[3], H);
	} else {
		model.rand_init(seed);
	}

	if(param.remove_bias) {
		double bias = training_set.get_global_mean();
		training_set.remove_bias(bias);
		test_set.remove_bias(bias);
		model.global_bias = bias;
	}

	// Random permutation for rows and cols of training_set for better load balancing
	std::vector<unsigned> row_perm, inverse_row_perm;
	std::vector<unsigned> col_perm, inverse_col_perm;
	if(do_shuffle) {
		gen_permutation_pair(training_set.rows, row_perm, inverse_row_perm);
		gen_permutation_pair(training_set.cols, col_perm, inverse_col_perm);

		training_set.apply_permutation(row_perm, col_perm);
		test_set.apply_permutation(row_perm, col_perm);
		if(nrhs >= 4)
			model.apply_permutation(inverse_row_perm, inverse_col_perm);
	}

	// Execute the program
	double time = omp_get_wtime();
	if(param.solver_type == SGD || param.solver_type == PU_SGD_ORIG)
		sgd(training_set, test_set, param, model);
	else if(param.solver_type == PU_SGD)
		sgd_pu(training_set, test_set, param, model);
	double walltime = omp_get_wtime() - time;

	if(do_shuffle) // recover the permutation for the model
		model.apply_permutation(row_perm, col_perm);

	// Write back the result
	plhs[0] = pmf_model_to_mxStruture(model);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*(mxGetPr(plhs[1])) = walltime;

	return 0;
} // }}}

// Interface function of matlab
// now assume prhs[0]: training_set, prhs[1]: test_set, prhs[2]: W, prhs[3]: H
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) { // {{{
	pmf_parameter_t param;

	// Transform the input Matrix to libsvm format
	if(nrhs >= 2 && nrhs <= 5)
	{
		if(!mxIsDouble(prhs[0])) {
			mexPrintf("Error: training matrix must be double\n");
			fake_answer(plhs);
			return;
		}

		if(!mxIsDouble(prhs[1])) {
			mexPrintf("Error: test matrix must be double\n");
			fake_answer(plhs);
			return;
		}

		param = parse_command_line(nrhs, prhs);
		switch (param.solver_type){
			case CCDR1:
			case CCDR1_SPEEDUP:
			case PU_CCDR1_SPEEDUP:
			case PU_CCDR1:
				run_ccdr1(plhs, nrhs, prhs, param);
				break;
			case ALS:
			case PU_ALS:
				run_als(plhs, nrhs, prhs, param);
				break;
			case SGD:
			case PU_SGD:
			case PU_SGD_ORIG:
				run_sgd(plhs, nrhs, prhs, param);
				break;
			default:
				fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
				exit_with_help();
				fake_answer(plhs);
				break;
		}
	} else {
		exit_with_help();
		fake_answer(plhs);
	}
} // }}}


