#include "mex.h"
#include "../pmf.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

typedef entry_t<val_type> ENTRY_T;
#define entry_t ENTRY_T

// Conversion Utility {{{
int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}
// convert matlab sparse matrix to C smat fmt
class mxSparse_iterator_t: public entry_iterator_t<val_type> {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols);
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		entry_t next() {
			while (cur_idx >= jc_t[cur_row+1])
				++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			entry_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt);
		}

};

// convert matlab Coo matrix to C smat fmt
class mxCoo_iterator_t: public entry_iterator_t<val_type> {
	private:
		double *row_idx, *col_idx, *val;
		size_t cur_idx;
		bool good_entry(size_t row, size_t col) {
			return 1 <= row && row <= rows && 1 <= col && col <= cols;
		}
		void init(double *_row_idx, double *_col_idx, double *_val, size_t _nnz, size_t _rows, size_t _cols) { // {{{
			row_idx = _row_idx; col_idx = _col_idx; val = _val; nnz = _nnz; rows = _rows; cols = _cols;
			cur_idx = 0;
			if(_rows == 0 && _cols == 0) {
				for(size_t idx = 0; idx < nnz; idx++) {
					if((size_t)row_idx[idx] > rows) rows = (size_t) row_idx[idx];
					if((size_t)col_idx[idx] > cols) cols = (size_t) col_idx[idx];
				}
			} else { // filter entries with out-of-range row/col indices
				for(size_t idx = 0; idx < nnz; idx++) {
					size_t row = (size_t) row_idx[idx], col = (size_t) col_idx[idx];
					if(!good_entry(row,col))
						nnz--;
				}
			}
		} // }}}

	public:
		size_t rows, cols;
		mxCoo_iterator_t(const mxArray *M, size_t _rows, size_t _cols) {
			double *data = mxGetPr(M);
			size_t _nnz = mxGetM(M);
			init(&data[0], &data[_nnz], &data[2*_nnz], _nnz, _rows, _cols);
		}
		mxCoo_iterator_t(double *_row_idx, double *_col_idx, double *_val, size_t _nnz, size_t _rows, size_t _cols) {
			init(_row_idx, _col_idx, _val, _nnz, _rows, _cols);
		}
		entry_t next() {
			size_t row = 0, col = 0;
			while(1) {
				row = (size_t) row_idx[cur_idx];
				col = (size_t) col_idx[cur_idx];
				if(good_entry(row, col))
					break;
				cur_idx++;
			}
			entry_t ret(row-1, col-1, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

// convert matlab Dense column-major matrix to C smat fmt
class mxDense_iterator_t: public entry_iterator_t<val_type> {
	private:
		size_t cur_idx;
		double *val;
	public:
		size_t rows, cols;
		mxDense_iterator_t(const mxArray *mxM): rows(mxGetM(mxM)), cols(mxGetN(mxM)), val(mxGetPr(mxM)){
			cur_idx = 0; nnz = rows*cols;
		}
		entry_t next() {
			entry_t ret(cur_idx%cols, cur_idx/cols, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

template<class T>
void mxSparse_to_smat(const mxArray *M, T &R) {
	size_t rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols);
	mxSparse_iterator_t entry_it(M);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
}

template<class T>
void mxCoo_to_smat(const mxArray *mxM, T &R, size_t rows=0, size_t cols=0) {
	mxCoo_iterator_t entry_it(mxM, rows, cols);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxCoo_to_smat(double *row_idx, double *col_idx, double *val, size_t nnz, T &R, size_t rows=0, size_t cols=0) {
	mxCoo_iterator_t entry_it(row_idx, col_idx, val, nnz, rows, cols);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxDense_to_smat(const mxArray *mxM, T &R) {
	mxDense_iterator_t entry_it(mxM);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
}

template<class T>
void mxArray_to_smat(const mxArray *mxM, T &R, size_t rows=0, size_t cols=0) {
	if(mxIsDouble(mxM) && mxIsSparse(mxM))
		mxSparse_to_smat(mxM, R);
	else if(mxIsDouble(mxM) && !mxIsSparse(mxM)) {
		mxCoo_to_smat(mxM, R, rows, cols);
	}
}
// }}} end-of-conversion

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) { // {{{
	size_t rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M.resize(cols, vec_t(rows,0));
	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			M[c][r] = val[idx++];
	return 0;
} // }}}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {// {{{
	size_t cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			val[idx++] = M[c][r];
	return 0;
} // }}}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) { // {{{
	size_t rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M.resize(rows, vec_t(cols,0));
	for(size_t c = 0, idx = 0; c < cols; c++)
		for(size_t r = 0; r < rows; r++)
			M[r][c] = val[idx++];
	return 0;
} // }}}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) { // {{{
	size_t rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(size_t c = 0, idx = 0; c < cols; ++c)
		for(size_t r = 0; r < rows; r++)
			val[idx++] = M[r][c];
	return 0;
} // }}}

mxArray* pmf_model_to_mxStruture(pmf_model_t& model) { // {{{
	static const char *field_names[] = {"W", "H", "global_bias"};
	static const int nr_fields = 3;
	mxArray *ret = mxCreateStructMatrix(1, 1, nr_fields, field_names);
	mxArray *mxW = mxCreateDoubleMatrix(model.rows, model.k, mxREAL);
	mxArray *mxH = mxCreateDoubleMatrix(model.cols, model.k, mxREAL);
	mxArray *mxglobal_bias = mxCreateDoubleMatrix(1, 1, mxREAL);
	if(model.major_type == pmf_model_t::COLMAJOR) {
		matCol_to_mxDense(model.W, mxW);
		matCol_to_mxDense(model.H, mxH);
	} else { // pmf_model_t::ROWMAJOR
		matRow_to_mxDense(model.W, mxW);
		matRow_to_mxDense(model.H, mxH);
	}
	*mxGetPr(mxglobal_bias) = model.global_bias;
	mxSetField(ret, 0, field_names[0], mxW);
	mxSetField(ret, 0, field_names[1], mxH);
	mxSetField(ret, 0, field_names[2], mxglobal_bias);
	return ret;
} // }}}

pmf_model_t gen_pmf_model(const mxArray *mxW, const mxArray *mxH,
		double global_bias=0, pmf_model_t::major_t major_type = pmf_model_t::ROWMAJOR) { // {{{
	size_t rows = mxGetM(mxW), cols = mxGetM(mxH), k = mxGetN(mxW);
	pmf_model_t model = pmf_model_t(rows, cols, k, major_type, false, global_bias);
	if(model.major_type == pmf_model_t::COLMAJOR) {
		mxDense_to_matCol(mxW, model.W);
		mxDense_to_matCol(mxH, model.H);
	} else { // pmf_model_t::ROWMAJOR
		mxDense_to_matRow(mxW, model.W);
		mxDense_to_matRow(mxH, model.H);
	}
	return model;
} // }}}

pmf_model_t mxStruture_to_pmf_model(const mxArray *mx_model, pmf_model_t::major_t major_type = pmf_model_t::ROWMAJOR) { // {{{
	static const char *field_names[] = {"W", "H", "global_bias"};
	static const int nr_fields = 3;
	mxArray *mxW = mxGetField(mx_model, 0, "W");
	mxArray *mxH = mxGetField(mx_model, 0, "H");
	mxArray *mxglobal_bias = mxGetField(mx_model, 0, "global_bias");
	double global_bias = *(mxGetPr(mxglobal_bias));
	return gen_pmf_model(mxW, mxH, global_bias, major_type);

} // }}}
