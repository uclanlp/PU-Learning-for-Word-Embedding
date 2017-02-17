#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>
#include <assert.h>
#include <omp.h>


#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8              int8_t;
typedef __int16             int16_t;
typedef __int32             int32_t;
typedef __int64             int64_t;
typedef unsigned __int8     uint8_t;
typedef unsigned __int16    uint16_t;
typedef unsigned __int32    uint32_t;
typedef unsigned __int64    uint64_t;
#endif
#endif


// Warning Copy Assignment missing


//#include "zlib_util.h"

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

#define dvec_t dense_vector
template<typename val_type> class dvec_t;
#define dmat_t dense_matrix
template<typename val_type> class dmat_t;
#define smat_t sparse_matrix
template<typename val_type> class smat_t;
template<typename val_type> class entry_iterator_t; // iterator for files with (i,j,v) tuples
template<typename val_type> class smat_iterator_t; // iterator for nonzero entries in smat_t
template<typename val_type> class smat_subset_iterator_t; // iterator for nonzero entries in a subset

// H = X*W, (X: m*n, W: n*k row-major, H m*k row major)
template<typename val_type> void smat_x_dmat(const smat_t<val_type> &X, const val_type* W, const size_t k, val_type *H);
template<typename val_type> void smat_x_dmat(const smat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H);
// H = a*X*W + H0, (X: m*n, W: n*k row-major, H m*k row major)
template<typename val_type> void smat_x_dmat(val_type a, const smat_t<val_type> &X, const val_type* W, const size_t k, const val_type *H0, val_type *H);
template<typename val_type> void smat_x_dmat(val_type a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H);

template<typename val_type>
class dvec_t{ // {{{
	public:
		size_t len;
	private:
		bool mem_alloc_by_me;
		val_type *buf;
	public:
		dvec_t(): len(0), buf(NULL), mem_alloc_by_me(false) {}
		dvec_t(const dvec_t& v): len(0), buf(0), mem_alloc_by_me(false) {*this=v;}
		dvec_t(size_t len, val_type *buf=NULL): len(len), buf(buf), mem_alloc_by_me(false){
			if(buf == NULL && len != 0) {
				this->buf = MALLOC(val_type, len);
				mem_alloc_by_me = true;
			}
		}
		~dvec_t() {if(mem_alloc_by_me) free(buf);}
		dvec_t& operator=(const dvec_t<val_type> &other) { // {{{
			if(this == &other)
				return *this;
			if(other.mem_alloc_by_me == false) { // shallow copy
				if(mem_alloc_by_me) { free(buf); buf = NULL;}
				len = other.len;
				buf = other.buf;
				mem_alloc_by_me = false;
			} else { // deep copy
				len = other.len;
				if(mem_alloc_by_me)
					buf = (val_type*) realloc(buf, sizeof(val_type)*len);
				else
					buf = MALLOC(val_type, len);
				memcpy(buf, other.buf, sizeof(val_type)*len);
				mem_alloc_by_me = true;
			}
			return *this;
		} // }}}
		size_t size() const {return len;};
		val_type& operator[](size_t idx) {return buf[idx];}
		const val_type& operator[](size_t idx) const {return buf[idx];}
		val_type* data() {return buf;}
		const val_type* data() const {return buf;}
}; // }}}

template<typename val_type>
class dmat_t{ // {{{
	public:
		size_t rows, cols;
	private:
		val_type *buf;
		bool mem_alloc_by_me;
		typedef dvec_t<val_type> vec_t;
		std::vector<vec_t> vec_set;
		void init_vec_set() { // {{{
			vec_set.resize(rows);
			for(size_t r = 0; r < rows; r++)
				vec_set[r] = vec_t(cols, &buf[r*cols]);
		} // }}}
	public:
		dmat_t(): rows(0), cols(0), buf(NULL), mem_alloc_by_me(false) {}
		dmat_t(const dmat_t& other): rows(0), cols(0), buf(NULL), mem_alloc_by_me(false){*this = other;}
		dmat_t(size_t rows, size_t cols, val_type *buf=NULL): rows(rows), cols(0), buf(buf), mem_alloc_by_me(false) { // {{{
			if(buf == NULL) {
				this->buf = MALLOC(val_type, rows*cols);
				mem_alloc_by_me = true;
			}
			init_vec_set();
		} // }}}
		dmat_t(size_t rows, const dvec_t<val_type>& v): rows(rows), cols(v.size()) { // {{{
			buf = MALLOC(val_type, rows*cols);
			mem_alloc_by_me = true;
			vec_set.resize(rows);
			for(size_t r = 0; r < rows; r++)
				vec_set[r] = vec_t(cols, &buf[r*cols]);
			init_vec_set();
		} // }}}
		~dmat_t() {if(mem_alloc_by_me) free(buf);}
		dmat_t& operator=(const dmat_t<val_type> &other) { // {{{
			if(this == &other)
				return *this;
			if(other.mem_alloc_by_me == false) { // shallow copy
				if(mem_alloc_by_me) { free(buf); buf=NULL;}
				rows = other.rows; cols = other.cols;
				buf = other.buf;
				init_vec_set();
				mem_alloc_by_me = false;
			} else { // deep copy
				resize(other.rows, other.cols);
				memcpy(buf, other.buf, sizeof(val_type)*rows*cols);
				mem_alloc_by_me = true;
			}
			return *this;
		} // }}}
		size_t size() const {return rows;}
		void resize(size_t rows_, const vec_t& v) { // {{{
			size_t cols_ = v.size();
			resize(rows_, cols_);
		} // }}}
		void resize(size_t rows_, size_t cols_) { // {{{
			if(mem_alloc_by_me) {
				if(rows_*cols_ != rows*cols)
					buf = (val_type*) realloc(buf, sizeof(val_type)*rows_*cols_);
			} else {
				buf = (val_type*) malloc(sizeof(val_type)*rows_*cols_);
			}
			mem_alloc_by_me = true;
			rows = rows_; cols = cols_;
			init_vec_set();
		} // }}}
		vec_t& operator[](size_t idx) {return vec_set[idx];}
		const vec_t& operator[](size_t idx) const {return vec_set[idx];}
		val_type* data() {return buf;}
		const val_type* data() const {return buf;}
}; // }}}

// Lapack and Blas support {{{
#ifdef _WIN32
#define ddot_ ddot
#define sdot_ sdot
#define daxpy_ daxpy
#define saxpy_ saxpy
#define dgemm_ dgemm
#define sgemm_ sgemm
#define dposv_ dposv
#define sposv_ sposv
#endif

extern "C" {

	double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	float sdot_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	ptrdiff_t saxpy_(ptrdiff_t *, float *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	double dcopy_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	float scopy_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	void dgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc);
	void sgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc);

	int dposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info);
	int sposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info);

}

template<typename val_type> val_type dot(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double dot(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return ddot_(len,x,xinc,y,yinc);}
template<> inline float dot(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return sdot_(len,x,xinc,y,yinc);}

template<typename val_type> ptrdiff_t axpy(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline ptrdiff_t axpy(ptrdiff_t *len, double *alpha, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return daxpy_(len,alpha,x,xinc,y,yinc);};
template<> inline ptrdiff_t axpy(ptrdiff_t *len, float *alpha, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return saxpy_(len,alpha,x,xinc,y,yinc);};

template<typename val_type> val_type copy(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double copy(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return dcopy_(len,x,xinc,y,yinc);}
template<> inline float copy(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return scopy_(len,x,xinc,y,yinc);}

template<typename val_type> void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, val_type *alpha, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, val_type *beta, val_type *c, ptrdiff_t *ldc);
template<> inline void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc) { dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
template<> inline void gemm<float>(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc) { sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

template<typename val_type> int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, ptrdiff_t *info);
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info) { return dposv_(uplo, n, nrhs, a, lda, b, ldb, info); }
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info) { return sposv_(uplo, n, nrhs, a, lda, b, ldb, info); }

// }}}

// <x,y>
template<typename val_type>
val_type do_dot_product(val_type *x, val_type *y, size_t size) { // {{{
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	return dot(&len, x, &inc, y, &inc);
} // }}}

// y = alpha*x + y
template<typename val_type>
void do_axpy(val_type alpha, val_type *x, val_type *y, size_t size) { // {{{
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	axpy(&len, &alpha, x, &inc, y, &inc);
} // }}}

// y = x
template<typename val_type>
void do_copy(val_type *x, val_type *y, size_t size) { // {{{
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	copy(&len, x, &inc, y, &inc);
} // }}}

// A, B, C are stored in column major!
template<typename val_type>
void dmat_x_dmat_colmajor(val_type alpha, val_type *A, bool trans_A, val_type *B, bool trans_B, val_type beta, val_type *C, size_t m, size_t n, size_t k) { // {{{
	ptrdiff_t mm = (ptrdiff_t)m, nn = (ptrdiff_t)n, kk = (ptrdiff_t)k;
	ptrdiff_t lda = trans_A? kk:mm, ldb = trans_B? nn:kk, ldc = mm;
	char transpose = 'T', notranspose = 'N';
	char *transa = trans_A? &transpose: &notranspose;
	char *transb = trans_B? &transpose: &notranspose;
	gemm(transa, transb, &mm, &nn, &kk, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
} // }}}

// C = alpha*A*B + beta*C
// C : m * n, k is the dimension of the middle
// A, B, C are stored in row major!
template<typename val_type>
void dmat_x_dmat(val_type alpha, val_type *A, bool trans_A, val_type *B, bool trans_B, val_type beta, val_type *C, size_t m, size_t n, size_t k) { // {{{
	dmat_x_dmat_colmajor(alpha, B, trans_B, A, trans_A, beta, C, n, m, k);
} //}}}

// C = A'*B
// C : m*n, k is the dimension of the middle
// A, B, C are stored in row major!
template<typename val_type>
void dmat_trans_x_dmat(val_type *A, val_type *B, val_type *C, size_t m, size_t n, size_t k) { // {{{
	bool trans = true; dmat_x_dmat(val_type(1.0), A, trans, B, !trans, val_type(0.0), C, m, n, k);
} // }}}

// C=A*B
// A, B, C are stored in row major!
template<typename val_type>
void dmat_x_dmat(val_type *A, val_type *B, val_type *C, size_t m, size_t n, size_t k) { // {{{
	bool trans = true; dmat_x_dmat(val_type(1.0), A, !trans, B, !trans, val_type(0.0), C, m, n, k);
} // }}}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
template<typename val_type>
void doHTH(val_type *H, val_type *HTH, size_t n, size_t k) { // {{{
	bool transpose = true;
	dmat_x_dmat_colmajor(val_type(1.0), H, !transpose, H, transpose, val_type(0.0), HTH, k, k, n);
} // }}}

// Solve Ax = b, A is symmetric positive definite, b is overwritten with the result x
// A will be modifed by internal Lapack. Make copy when necessary
template<typename val_type>
bool ls_solve_chol(val_type *A, val_type *b, int n) { // {{{
  ptrdiff_t nn=n, lda=n, ldb=n, nrhs=1, info;
  char uplo = 'U';
  posv(&uplo, &nn, &nrhs, A, &lda, b, &ldb, &info);
  return (info == 0);
} // }}}


// Sparse matrix format CSC & CSR
template<typename val_type>
class smat_t{
	private:
		bool mem_alloc_by_me;
		bool read_from_binary;
		unsigned char* binary_buf;
		size_t binary_buf_len;
		const static int HeaderSize =
			sizeof(size_t)+sizeof(size_t)+sizeof(size_t)+sizeof(size_t);
		void csr_to_csc();
		void csc_to_csr();
	public:
		size_t rows, cols;
		size_t nnz, max_row_nnz, max_col_nnz;
		val_type *val, *val_t;
		size_t *col_ptr, *row_ptr;
		unsigned *row_idx, *col_idx;

		// filetypes for loading smat_t
		enum format_t {TXT=0, PETSc=1, BINARY=2, COMPRESSION=3};

		// Constructor and Destructor
		smat_t() : mem_alloc_by_me(false), read_from_binary(false), rows(0), cols(0), nnz(0){
		val=val_t=NULL; col_ptr=row_ptr=NULL, row_idx=col_idx=NULL;}
		smat_t(const smat_t& m){*this = m; mem_alloc_by_me = false; read_from_binary = false;}
		~smat_t(){ clear_space();}

		void clear_space();
		smat_t transpose();
		const smat_t transpose() const;
		void apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm);
		void apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL);
		smat_subset_iterator_t<val_type> row_subset_it(const std::vector<unsigned> &subset);
		smat_subset_iterator_t<val_type> row_subset_it(const unsigned *subset, int subset_size);
		smat_subset_iterator_t<val_type> col_subset_it(const std::vector<unsigned> &subset);
		smat_subset_iterator_t<val_type> col_subset_it(const unsigned *subset, int subset_size);
		smat_t row_subset(const std::vector<unsigned> &subset);
		smat_t row_subset(const unsigned *subset, int subset_size);

		size_t nnz_of_row(unsigned i) const {return (row_ptr[i+1]-row_ptr[i]);}
		size_t nnz_of_col(unsigned i) const {return (col_ptr[i+1]-col_ptr[i]);}

		// smat-vector multiplication
		void Xv(const val_type *v, val_type *Xv);
		void XTu(const val_type *u, val_type *XTu);

		// IO methods
		void load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t<val_type>* entry_it);
		void load(size_t _rows, size_t _cols, size_t _nnz, const char *filename, format_t fmt);
		void load_from_PETSc(const char  *filename);
		void save_PETSc_to_file(const char *filename);
		void load_from_binary(const char *filename);
		void save_binary_to_file(const char *filename);

		// used for MPI verions
		void from_mpi(){
			mem_alloc_by_me = true;
			max_col_nnz = 0;
			for(size_t c = 0; c < cols; c++)
				max_col_nnz = std::max(max_col_nnz, nnz_of_col(c));
		}
		val_type get_global_mean() const;
		void remove_bias(val_type bias=0);
};


/*-------------- Iterators -------------------*/

template<typename val_type>
class entry_t{
	public:
		unsigned i, j; val_type v, weight;
		entry_t(int ii=0, int jj=0, val_type vv=0, val_type ww=1.0): i(ii), j(jj), v(vv), weight(ww){}
};

template<typename val_type>
class entry_iterator_t {
	public:
		size_t nnz;
		virtual entry_t<val_type> next() = 0;
};

#define MAXLINE 10240
// Iterator for files with (i,j,v) tuples
template<typename val_type>
class file_iterator_t: public entry_iterator_t<val_type>{
	public:
		file_iterator_t(size_t nnz_, const char* filename, size_t start_pos=0);
		~file_iterator_t(){ if (fp) fclose(fp); }
		entry_t<val_type> next();
	private:
		size_t nnz;
		FILE *fp;
		char line[MAXLINE];
};

// smat_t iterator
template<typename val_type>
class smat_iterator_t: public entry_iterator_t<val_type>{
	public:
		enum {ROWMAJOR, COLMAJOR};
		// major: smat_iterator_t<val_type>::ROWMAJOR or smat_iterator_t<val_type>::COLMAJOR
		smat_iterator_t(const smat_t<val_type>& M, int major = ROWMAJOR);
		~smat_iterator_t() {}
		entry_t<val_type> next();
	private:
		size_t nnz;
		unsigned *col_idx;
		size_t *row_ptr;
		val_type *val_t;
		size_t rows, cols, cur_idx;
		size_t cur_row;
};

// smat_t subset iterator
template<typename val_type>
class smat_subset_iterator_t: public entry_iterator_t<val_type>{
	public:
		enum {ROWMAJOR, COLMAJOR};
		// major: smat_iterator_t<val_type>::ROWMAJOR or smat_iterator_t<val_type>::COLMAJOR
		smat_subset_iterator_t(const smat_t<val_type>& M, const unsigned *subset, size_t size, bool remapping=false, int major = ROWMAJOR);
		~smat_subset_iterator_t() {}
		size_t get_nnz() {return nnz;}
		size_t get_rows() {return major==ROWMAJOR? remapping? subset.size(): rows: rows;}
		size_t get_cols() {return major==ROWMAJOR? cols: remapping? subset.size():cols;}
		entry_t<val_type> next();
	private:
		size_t nnz;
		unsigned *col_idx;
		size_t *row_ptr;
		val_type *val_t;
		size_t rows, cols, cur_idx;
		size_t cur_row;
		std::vector<unsigned>subset;
		int major;
		bool remapping;
};

// -------------- Implementation --------------
template<typename val_type>
void smat_t<val_type>::clear_space() {
	if(mem_alloc_by_me) {
		if(read_from_binary)
			free(binary_buf);
		else {
			if(val)free(val); if(val_t)free(val_t);
			if(row_ptr)free(row_ptr);if(row_idx)free(row_idx);
			if(col_ptr)free(col_ptr);if(col_idx)free(col_idx);
		}
	}
	read_from_binary = false;
	mem_alloc_by_me = false;
}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::transpose(){
	smat_t<val_type> mt;
	mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
	mt.val = val_t; mt.val_t = val;
	mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
	mt.col_idx = row_idx; mt.row_idx = col_idx;
	mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
	return mt;
}
template<typename val_type>
const smat_t<val_type> smat_t<val_type>::transpose() const {
	return transpose();
}

template<typename val_type>
void smat_t<val_type>::apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) {
	apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0]: NULL);
}

template<typename val_type>
void smat_t<val_type>::apply_permutation(const unsigned *row_perm, const unsigned *col_perm) {
	if(row_perm!=NULL) {
		for(size_t idx = 0; idx < nnz; idx++) row_idx[idx] = row_perm[row_idx[idx]];
		csc_to_csr();
		csr_to_csc();
	}
	if(col_perm!=NULL) {
		for(size_t idx = 0; idx < nnz; idx++) col_idx[idx] = col_perm[col_idx[idx]];
		csr_to_csc();
		csc_to_csr();
	}
}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::row_subset_it(const std::vector<unsigned> &subset) {
	return row_subset_it(&subset[0], (int)subset.size());
}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::row_subset_it(const unsigned *subset, int subset_size) {
	return smat_subset_iterator_t<val_type> (*this, subset, subset_size);
}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::col_subset_it(const std::vector<unsigned> &subset) {
	return col_subset_it(&subset[0], (int)subset.size());
}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::col_subset_it(const unsigned *subset, int subset_size) {
	bool remmapping = false; // no remapping by default
	return smat_subset_iterator_t<val_type> (*this, subset, subset_size, remmapping, smat_subset_iterator_t<val_type>::COLMAJOR);
}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::row_subset(const std::vector<unsigned> &subset) {
	return row_subset(&subset[0], (int)subset.size());
}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::row_subset(const unsigned *subset, int subset_size) {
	smat_subset_iterator_t<val_type> it(*this, subset, subset_size);
	smat_t<val_type> sub_smat;
	sub_smat.load_from_iterator(subset_size, cols, it.get_nnz(), &it);
	return sub_smat;
}

template<typename val_type>
val_type smat_t<val_type>::get_global_mean() const {
	val_type sum=0;
	for(size_t idx = 0; idx < nnz; idx++) sum += val[idx];
	return sum/(val_type)nnz;
}

template<typename val_type>
void smat_t<val_type>::remove_bias(val_type bias){
	if(bias) {
		for(size_t idx = 0; idx < nnz; idx++) {
			val[idx] -= bias;
			val_t[idx] -= bias;
		}
	}
}

template<typename val_type>
void smat_t<val_type>::Xv(const val_type *v, val_type *Xv) {
	for(size_t i = 0; i < rows; ++i) {
		Xv[i] = 0;
		for(size_t idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx)
			Xv[i] += val_t[idx] * v[col_idx[idx]];
	}
}

template<typename val_type>
void smat_t<val_type>::XTu(const val_type *u, val_type *XTu) {
	for(size_t i = 0; i < cols; ++i) {
		XTu[i] = 0;
		for(size_t idx = col_ptr[i]; idx < col_ptr[i+1]; ++idx)
			XTu[i] += val[idx] * u[row_idx[idx]];
	}
}

// Comparator for sorting rates into row/column comopression storage
template<typename val_type>
class SparseComp {
	public:
		const unsigned *row_idx;
		const unsigned *col_idx;
		SparseComp(const unsigned *row_idx_, const unsigned *col_idx_, bool isCSR=true) {
			row_idx = (isCSR)? row_idx_: col_idx_;
			col_idx = (isCSR)? col_idx_: row_idx_;
		}
		bool operator()(size_t x, size_t y) const {
			return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]< col_idx[y]));
		}
};

template<typename val_type>
void smat_t<val_type>::load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t<val_type> *entry_it){
	clear_space(); // clear any pre-allocated space in case of memory leak
	rows =_rows,cols=_cols,nnz=_nnz;
	mem_alloc_by_me = true;
	val = MALLOC(val_type, nnz); val_t = MALLOC(val_type, nnz);
	row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);
	//row_idx = MALLOC(unsigned long, nnz); col_idx = MALLOC(unsigned long, nnz); // switch to this for matlab
	row_ptr = MALLOC(size_t, rows+1); col_ptr = MALLOC(size_t, cols+1);
	memset(row_ptr,0,sizeof(size_t)*(rows+1));
	memset(col_ptr,0,sizeof(size_t)*(cols+1));

	// a trick here to utilize the space the have been allocated
	std::vector<size_t> perm(_nnz);
	unsigned *tmp_row_idx = col_idx;
	unsigned *tmp_col_idx = row_idx;
	val_type *tmp_val = val;
	for(size_t idx = 0; idx < _nnz; idx++){
		entry_t<val_type> rate = entry_it->next();
		row_ptr[rate.i+1]++;
		col_ptr[rate.j+1]++;
		tmp_row_idx[idx] = rate.i;
		tmp_col_idx[idx] = rate.j;
		tmp_val[idx] = rate.v;
		perm[idx] = idx;
	}
	// sort entries into row-majored ordering
	sort(perm.begin(), perm.end(), SparseComp<val_type>(tmp_row_idx, tmp_col_idx, true));
	// Generate CSR format
	for(size_t idx = 0; idx < _nnz; idx++) {
		val_t[idx] = tmp_val[perm[idx]];
		col_idx[idx] = tmp_col_idx[perm[idx]];
	}

	// Calculate nnz for each row and col
	max_row_nnz = max_col_nnz = 0;
	for(size_t r = 1; r <= rows; r++) {
		max_row_nnz = std::max(max_row_nnz, row_ptr[r]);
		row_ptr[r] += row_ptr[r-1];
	}
	for(size_t c = 1; c <= cols; c++) {
		max_col_nnz = std::max(max_col_nnz, col_ptr[c]);
		col_ptr[c] += col_ptr[c-1];
	}

	// Transpose CSR into CSC matrix
	for(size_t r = 0; r < rows; ++r){
		for(size_t idx = row_ptr[r]; idx < row_ptr[r+1]; idx++){
			size_t c = (size_t) col_idx[idx];
			row_idx[col_ptr[c]] = r;
			val[col_ptr[c]++] = val_t[idx];
		}
	}
	for(size_t c = cols; c > 0; --c) col_ptr[c] = col_ptr[c-1];
	col_ptr[0] = 0;
}

template<typename val_type>
void smat_t<val_type>::load(size_t _rows, size_t _cols, size_t _nnz, const char* filename, typename smat_t<val_type>::format_t fmt){

	if(fmt == smat_t<val_type>::TXT) {
		file_iterator_t<val_type> entry_it(_nnz, filename);
		load_from_iterator(_rows, _cols, _nnz, &entry_it);
	} else if(fmt == smat_t<val_type>::PETSc) {
		load_from_PETSc(filename);
	} else {
		fprintf(stderr, "Error: filetype %d not supported\n", fmt);
		return ;
	}
}

template<typename val_type>
void smat_t<val_type>::save_PETSc_to_file(const char *filename){
	const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
	FILE *fp = fopen(filename, "wb");
	if(fp == NULL) {
		fprintf(stderr,"Error: can't open file %s\n", filename);
		exit(1);
	}
	int32_t int_buf[3] = {(int32_t)LONG_FILE, (int32_t)rows, (int32_t)cols};
	std::vector<int32_t> nnz_row(rows);
	for(size_t r = 0; r < rows; r++)
		nnz_row[r] = (int)nnz_of_row(r);

	fwrite(&int_buf[0], sizeof(int32_t), 3, fp);
	fwrite(&nnz, sizeof(size_t), 1, fp);
	fwrite(&nnz_row[0], sizeof(int32_t), rows, fp);
	fwrite(&col_idx[0], sizeof(unsigned), nnz, fp);

	// the following part == fwrite(val_t, sizeof(double), nnz, fp);
	const size_t chunksize = 1024;
	double buf[chunksize];
	size_t idx = 0;
	while(idx + chunksize < nnz) {
		for(size_t i = 0; i < chunksize; i++)
			buf[i] = (double) val_t[idx+i];
		fwrite(&buf[0], sizeof(double), chunksize, fp);
		idx += chunksize;
	}
	size_t remaining = nnz - idx;
	for(size_t i = 0; i < remaining; i++)
		buf[i] = (double) val_t[idx+i];
	fwrite(&buf[0], sizeof(double), remaining, fp);

	fclose(fp);
}

template<typename val_type>
void smat_t<val_type>::load_from_PETSc(const char *filename) {
	clear_space(); // clear any pre-allocated space in case of memory leak
	const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
	int32_t int_buf[3];
	size_t headersize = 0;
	FILE *fp = fopen(filename, "rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
		return;
	}
	headersize += sizeof(int)*fread(int_buf, sizeof(int), 3, fp);
	int filetype = int_buf[0];
	rows = (size_t) int_buf[1];
	cols = (size_t) int_buf[2];
	if(filetype == UNSIGNED_FILE) {
		headersize += sizeof(int)*fread(int_buf, sizeof(int32_t), 1, fp);
		nnz = (size_t) int_buf[0];
	} else if (filetype == LONG_FILE){
		headersize += sizeof(size_t)*fread(&nnz, sizeof(int64_t), 1, fp);
	} else {
		fprintf(stderr, "Error: wrong PETSc format for %s\n", filename);
	}
	// Allocation of memory
	mem_alloc_by_me = true;
	val = MALLOC(val_type, nnz); val_t = MALLOC(val_type, nnz);
	row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);
	row_ptr = MALLOC(size_t, rows+1); col_ptr = MALLOC(size_t, cols+1);

	// load CSR from the binary PETSc format
	{
		// read row_ptr
		std::vector<int32_t> nnz_row(rows);
		headersize += sizeof(int32_t)*fread(&nnz_row[0], sizeof(int32_t), rows, fp);
		row_ptr[0] = 0;
		for(size_t r = 1; r <= rows; r++)
			row_ptr[r] = row_ptr[r-1] + nnz_row[r-1];
		// read col_idx
		headersize += sizeof(int)*fread(&col_idx[0], sizeof(unsigned), nnz, fp);

		// read val_t
		const size_t chunksize = 1024;
		double buf[chunksize];
		size_t idx = 0;
		while(idx + chunksize < nnz) {
			headersize += sizeof(double)*fread(&buf[0], sizeof(double), chunksize, fp);
			for(size_t i = 0; i < chunksize; i++)
				val_t[idx+i] = (val_type) buf[i];
			idx += chunksize;
		}
		size_t remaining = nnz - idx;
		headersize += sizeof(double)*fread(&buf[0], sizeof(double), remaining, fp);
		for(size_t i = 0; i < remaining; i++)
			val_t[idx+i] = (val_type) buf[i];
	}
	fclose(fp);

	csr_to_csc(); // Convert CSR to CSC
	max_row_nnz = max_col_nnz = 0;
	for(size_t c = 0; c < cols; c++) max_col_nnz = std::max(max_col_nnz, nnz_of_col(c));
	for(size_t r = 0; r < rows; r++) max_row_nnz = std::max(max_row_nnz, nnz_of_row(r));
}

template<typename val_type>
void smat_t<val_type>::csr_to_csc() {
	memset(col_ptr, 0, sizeof(size_t)*(cols+1));
	for(size_t idx = 0; idx < nnz; idx++)
		col_ptr[col_idx[idx]+1]++;
	for(size_t c = 1; c <= cols; c++)
		col_ptr[c] += col_ptr[c-1];
	for(size_t r = 0; r < rows; r++) {
		for(size_t idx = row_ptr[r]; idx != row_ptr[r+1]; idx++) {
			size_t c = (size_t) col_idx[idx];
			row_idx[col_ptr[c]] = r;
			val[col_ptr[c]++] = val_t[idx];
		}
	}
	for(size_t c = cols; c > 0; c--)
		col_ptr[c] = col_ptr[c-1];
	col_ptr[0] = 0;
}

template<typename val_type>
void smat_t<val_type>::csc_to_csr() {
	memset(row_ptr, 0, sizeof(size_t)*(rows+1));
	for(size_t idx = 0; idx < nnz; idx++)
		row_ptr[row_idx[idx]+1]++;
	for(size_t r = 1; r <= rows; r++)
		row_ptr[r] += row_ptr[r-1];
	for(size_t c = 0; c < cols; c++) {
		for(size_t idx = col_ptr[c]; idx != col_ptr[c+1]; idx++) {
			size_t r = (size_t) row_idx[idx];
			col_idx[row_ptr[r]] = c;
			val_t[row_ptr[r]++] = val[idx];
		}
	}
	for(size_t r = rows; r > 0; r--)
		row_ptr[r] = row_ptr[r-1];
	row_ptr[0] = 0;
}

template<typename val_type>
file_iterator_t<val_type>::file_iterator_t(size_t nnz_, const char* filename, size_t start_pos){
	nnz = nnz_;
	fp = fopen(filename,"rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: cannot read the file (%s)!!\n", filename);
		return;
	}
	fseek(fp, start_pos, SEEK_SET);
}

template<typename val_type>
entry_t<val_type> file_iterator_t<val_type>::next() {
	const int base10 = 10;
	if(nnz > 0) {
		--nnz;
		if(fgets(&line[0], MAXLINE, fp)==NULL)
			fprintf(stderr, "Error: reading error !!\n");
		char *head_ptr = &line[0];
		size_t i = strtol(head_ptr, &head_ptr, base10);
		size_t j = strtol(head_ptr, &head_ptr, base10);
		double v = strtod(head_ptr, &head_ptr);
		return entry_t<val_type>(i-1, j-1, (val_type)v);
	} else {
		fprintf(stderr, "Error: no more entry to iterate !!\n");
		return entry_t<val_type>(0,0,0);
	}
}
/* Deprecated Implementation
template<typename val_type>
entry_t<val_type> file_iterator_t<val_type>::next() {
	int i = 1, j = 1;
	val_type v = 0;
	if (nnz > 0) {
#ifdef _USE_FLOAT_
		if(fscanf(fp, "%d %d %f", &i, &j, &v)!=3)
			fprintf(stderr, "Error: reading smat_t\n");
#else
		if(fscanf(fp, "%d %d %lf", &i, &j, &v)!=3)
			fprintf(stderr, "Error: reading smat_t\n");
#endif
		--nnz;
	} else {
		fprintf(stderr,"Error: no more entry to iterate !!\n");
	}
	return entry_t<val_type>(i-1,j-1,v);
}
*/

template<typename val_type>
smat_iterator_t<val_type>::smat_iterator_t(const smat_t<val_type>& M, int major) {
	nnz = M.nnz;
	col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
	row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
	val_t = (major == ROWMAJOR)? M.val_t: M.val;
	rows = (major==ROWMAJOR)? M.rows: M.cols;
	cols = (major==ROWMAJOR)? M.cols: M.rows;
	cur_idx = cur_row = 0;
}

template<typename val_type>
entry_t<val_type> smat_iterator_t<val_type>::next() {
	while (cur_idx >= row_ptr[cur_row+1])
		cur_row++;
	if (nnz > 0)
		nnz--;
	else
		fprintf(stderr,"Error: no more entry to iterate !!\n");
	entry_t<val_type> ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
	cur_idx++;
	return ret;
}

template<typename val_type>
smat_subset_iterator_t<val_type>::smat_subset_iterator_t(const smat_t<val_type>& M, const unsigned *subset, size_t size, bool remapping_, int major_) {
	major = major_; remapping = remapping_;
	col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
	row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
	val_t = (major == ROWMAJOR)? M.val_t: M.val;
	rows = (major==ROWMAJOR)? (remapping?size:M.rows): (remapping?size:M.cols);
	cols = (major==ROWMAJOR)? M.cols: M.rows;
	this->subset.resize(size);
	nnz = 0;
	for(size_t i = 0; i < size; i++) {
		unsigned idx = subset[i];
		this->subset[i] = idx;
		nnz += (major == ROWMAJOR)? M.nnz_of_row(idx): M.nnz_of_col(idx);
	}
	sort(this->subset.begin(), this->subset.end());
	cur_row = 0;
	cur_idx = row_ptr[this->subset[cur_row]];
}

template<typename val_type>
entry_t<val_type> smat_subset_iterator_t<val_type>::next() {
	while (cur_idx >= row_ptr[subset[cur_row]+1]) {
		cur_row++;
		cur_idx = row_ptr[subset[cur_row]];
	}
	if (nnz > 0)
		nnz--;
	else
		fprintf(stderr,"Error: no more entry to iterate !!\n");
	//entry_t<val_type> ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
	entry_t<val_type> ret_rowwise(remapping?cur_row:subset[cur_row], col_idx[cur_idx], val_t[cur_idx]);
	entry_t<val_type> ret_colwise(col_idx[cur_idx], remapping?cur_row:subset[cur_row], val_t[cur_idx]);
	//printf("%d %d\n", cur_row, col_idx[cur_idx]);
	cur_idx++;
	//return ret;
	return major==ROWMAJOR? ret_rowwise: ret_colwise;
}


/*
   H = X*W
   X is an m*n
   W is an n*k, row-majored array
   H is an m*k, row-majored array
   */
template<typename val_type>
void smat_x_dmat(const smat_t<val_type> &X, const val_type* W, const size_t k, val_type *H) {
	size_t m = X.rows;
#pragma omp parallel for schedule(dynamic,50) shared(X,H,W)
	for(size_t i = 0; i < m; i++) {
		val_type *Hi = &H[k*i];
		memset(Hi,0,sizeof(val_type)*k);
		for(size_t idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
			const val_type Xij = X.val_t[idx];
			const val_type *Wj = &W[X.col_idx[idx]*k];
			for(unsigned t = 0; t < k; t++)
				Hi[t] += Xij*Wj[t];
		}
	}
}
template<typename val_type>
void smat_x_dmat(const smat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H) {
	assert(W.cols == H.cols && X.cols == W.rows && X.rows == H.rows);
	smat_x_dmat(X, W.data(), W.cols, H.data());
}


/*
   H = a*X*W + H0
   X is an m*n
   W is an n*k, row-majored array
   H is an m*k, row-majored array
   */
template<typename val_type>
void smat_x_dmat(val_type a, const smat_t<val_type> &X, const val_type* W, const size_t k, const val_type *H0, val_type *H) {
	size_t m = X.rows;
#pragma omp parallel for schedule(dynamic,50) shared(X,H,W)
	for(size_t i = 0; i < m; i++) {
		val_type *Hi = &H[k*i];
		if(H != H0)
			memcpy(Hi, &H0[k*i], sizeof(val_type)*k);
		for(size_t idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
			const val_type Xij = X.val_t[idx];
			const val_type *Wj = &W[X.col_idx[idx]*k];
			for(unsigned t = 0; t < k; t++)
				Hi[t] += a*Xij*Wj[t];
		}
	}
}
template<typename val_type>
void smat_x_dmat(val_type a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
	assert(W.cols == H0.cols && W.cols == H.cols && X.cols == W.rows && X.rows == H0.rows && X.rows == H.rows);
	smat_x_dmat(a, X, W.data(), W.cols, H0.data(), H.data());
}

#undef smat_t
#undef dmat_t
#undef dvec_t
#endif // SPARSE_MATRIX_H
