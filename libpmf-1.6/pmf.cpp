#include "pmf.h"

pmf_model_t::pmf_model_t(size_t rows_, size_t cols_, size_t k_, major_t major_type_, bool do_rand_init, val_type global_bias_){
	rows = rows_;
	cols = cols_;
	k = k_;
	major_type = major_type_;
	if(do_rand_init)
		rand_init();
	global_bias = global_bias_;
}

#ifdef _MSC_VER
#define srand48(seed) std::srand(seed)
#define drand48() ((double)std::rand()/(RAND_MAX+1))
#endif

void pmf_model_t::mat_rand_init(mat_t &X, size_t m, size_t n, long seed) {
	val_type scale = 1./sqrt(k);
	rng_t rng(seed);
	//srand48(seed);
	if(major_type == COLMAJOR) {
		X.resize(n, vec_t(m));
		for(size_t i = 0; i < m; i++)
			for(size_t j = 0; j < n; j++)
				X[j][i] = (val_type) rng.uniform((val_type)0.0, scale);
		//		X[j][i] = (val_type) (scale*(2*drand48()-1.0));
	    //		X[j][i] = (val_type) (scale*(drand48()));
	} else { // major_type == ROWMAJOR
		X.resize(m, vec_t(n));
		for(size_t i = 0; i < m; i++)
			for(size_t j = 0; j < n; j++)
				X[i][j] = (val_type) rng.uniform((val_type)0.0, scale);
	//			X[i][j] = (val_type) (scale*(2*drand48()-1.0));
	//          X[i][j] = (val_type) (scale*(drand48()));
	}
}

void pmf_model_t::rand_init(long seed) {
	mat_rand_init(W, rows, k, seed);
	mat_rand_init(H, cols, k, seed+k);
}

val_type pmf_model_t::predict_entry(size_t i, size_t j) const {
	val_type value = global_bias;
	if(0 <= i && i < rows && 0 <= j && j < cols) {
		if(major_type == COLMAJOR) {
			for(size_t t = 0; t < k; t++)
				value += W[t][i] * H[t][j];
		} else { // major_type == ROWMAJOR
			const vec_t &Wi = W[i], &Hj = H[j];
			for(size_t t = 0; t < k; t++)
				value += Wi[t] * Hj[t];
		}
	}
	return value;
}

void pmf_model_t::apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) {
	apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0] : NULL);
}
void pmf_model_t::apply_permutation(const unsigned *row_perm, const unsigned *col_perm) {

	if(major_type == COLMAJOR) {
		vec_t u(rows), v(cols);
		for(size_t t = 0; t < k; t++) {
			vec_t &Wt = W[t], &Ht = H[t];
			if(row_perm != NULL) {
				for(size_t r = 0; r < rows; r++)
					u[r] = Wt[r];
				for(size_t r = 0; r < rows; r++)
					Wt[r] = u[row_perm[r]];
			}
			if(col_perm != NULL) {
				for(size_t c = 0; c < cols; c++)
					v[c] = Ht[c];
				for(size_t c = 0; c < cols; c++)
					Ht[c] = v[col_perm[c]];
			}
		}
	} else { // major_type == ROWMAJOR
		if(row_perm != NULL) {
			mat_t buf(rows, vec_t(k));
			for(size_t r = 0; r < rows; r++)
				for(size_t t = 0; t < k; t++)
					buf[r][t] = W[r][t];

			for(size_t r = 0; r < rows; r++)
				for(size_t t = 0; t < k; t++)
					W[r][t] = buf[row_perm[r]][t];
		}
		if(col_perm != NULL) {
			mat_t buf(cols, vec_t(k));
			for(size_t c = 0; c < cols; c++)
				for(size_t t = 0; t < k; t++)
					buf[c][t] = H[c][t];
			for(size_t c = 0; c < cols; c++)
				for(size_t t = 0; t < k; t++)
					H[c][t] = buf[col_perm[c]][t];
		}
	}
}

void pmf_model_t::save(FILE *fp){
	save_mat_t(W, fp, major_type==ROWMAJOR);
	save_mat_t(H, fp, major_type==ROWMAJOR);
	double buf = (double)global_bias;
	fwrite(&buf, sizeof(double), 1, fp);
}
void pmf_model_t::save_embedding(FILE *fp){
    printf("???");
    save_wordembedding(W, H, fp, major_type==ROWMAJOR);
    // Need to save the embedding in ascii file an compare
    // Check how save_mat_t is implemented

}

void pmf_model_t::load(FILE *fp, major_t major_type_){
	major_type = major_type_;
	load_mat_t(fp, W, major_type==ROWMAJOR);
	load_mat_t(fp, H, major_type==ROWMAJOR);
	double buf = 0;
	if(fread(&buf, sizeof(double), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	global_bias = (val_type) buf;
	rows = (major_type==ROWMAJOR)? W.size() : W[0].size();
	cols = (major_type==ROWMAJOR)? H.size() : H[0].size();
	k = (major_type==ROWMAJOR)? W[0].size() : W.size();
}

// Save a mat_t A to a file in row_major order.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(const mat_t &A, FILE *fp, bool row_major){//{{{
	if (fp == NULL)
		fprintf(stderr, "output stream is not valid.\n");
	long m = row_major? A.size(): A[0].size();
	long n = row_major? A[0].size(): A.size();
	fwrite(&m, sizeof(long), 1, fp);
	fwrite(&n, sizeof(long), 1, fp);
	double *buf = MALLOC(double, m*n);

	if (row_major) {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
            	buf[idx++] = A[i][j];
	} else {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
		buf[idx++] = A[j][i];
	}
	fwrite(&buf[0], sizeof(double), m*n, fp);
    

	fflush(fp);
	free(buf);
}//}}}

void save_wordembedding(const mat_t &A,  const  mat_t &B, FILE *fp, bool row_major){//{{{
	if (fp == NULL)
		fprintf(stderr, "output stream is not valid.\n");
	long m = row_major? A.size(): A[0].size();
	long n = row_major? A[0].size(): A.size();
	double *buf = MALLOC(double, m*n);

	if (row_major) {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
        {
			for(size_t j = 0; j < n; ++j)
                fprintf(fp, "%.6lf ", A[i][j],B[i][j]);//FIXIT
            fprintf(fp, "\n");
        }
    } else {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
        {
			for(size_t j = 0; j < n; ++j)
                fprintf(fp, "%.6lf ", A[j][i],B[j][i]);//FIXIT
            fprintf(fp, "\n");
        }

    }
	fflush(fp);
}//}}}



// Load a matrix from a file and return a mat_t matrix
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returened A  is stored in col_major order.
void load_mat_t(FILE *fp, mat_t &A, bool row_major){//{{{
	if (fp == NULL)
		fprintf(stderr, "Error: null input stream.\n");
	long m, n;
	if(fread(&m, sizeof(long), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	if(fread(&n, sizeof(long), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	double *buf = MALLOC(double, m*n);
	if(fread(&buf[0], sizeof(val_type), m*n, fp) != m*n)
		fprintf(stderr, "Error: wrong input stream.\n");
	if (row_major) {
		//A = mat_t(m, vec_t(n));
		A.resize(m, vec_t(n));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[i][j] = buf[idx++];
	} else {
		//A = mat_t(n, vec_t(m));
		A.resize(n, vec_t(m));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[j][i] = buf[idx++];
	}
	free(buf);
}//}}}


// load utility for CCS RCS
void pmf_read_data(const char* srcdir, smat_t &training_set, smat_t &test_set, smat_t::format_t fmt) { //{{{
	size_t m, n, nnz;
	char filename[1024], buf[1024], suffix[12];
	FILE *fp;
	sprintf(filename,"%s/meta",srcdir);
	fp = fopen(filename,"r");
	if(fscanf(fp, "%lu %lu", &m, &n) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
		return;
	}

	if(fscanf(fp, "%lu %s", &nnz, buf) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
		return;
	}
	if(fmt == smat_t::TXT)
		suffix[0] = 0; //sprintf(suffix, "");
	else if(fmt == smat_t::PETSc)
		sprintf(suffix, ".petsc");
	else
		printf("Error: fmt %d is not supported.", fmt);

	sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
	training_set.load(m, n, nnz, filename, fmt);

	if(fscanf(fp, "%lu %s", &nnz, buf) != EOF){
		sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
		test_set.load(m, n, nnz, filename, fmt);
	}
	fclose(fp);
	return ;
}//}}}

// load utility for blocks_t
void pmf_read_data(const char* srcdir, blocks_t &training_set, blocks_t &test_set, smat_t::format_t fmt) { //{{{
	size_t m, n, nnz;
	char filename[1024], buf[1024], suffix[12];
	FILE *fp;
	sprintf(filename,"%s/meta",srcdir);
	fp = fopen(filename,"r");
	if(fscanf(fp, "%lu %lu", &m, &n) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
		return;
	}

	if(fscanf(fp, "%lu %s", &nnz, buf) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
		return;
	}
	suffix[0] = 0; // TXT
	if(fmt == smat_t::TXT)
		suffix[0] = 0; //sprintf(suffix, "");
	else if(fmt == smat_t::PETSc)
		sprintf(suffix, ".petsc");
	else
		printf("Error: fmt %d is not supported.", fmt);

	sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
	training_set.load(m, n, nnz, filename, fmt);
	//training_set.load(m, n, nnz, filename);

	if(fscanf(fp, "%lu %s", &nnz, buf) != EOF){
		sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
		test_set.load(m, n, nnz, filename, fmt);
		//test_set.load(m, n, nnz, filename);
	}
	fclose(fp);
	return ;
}//}}}


// Ranking Evaluation Utlility Function

struct decreasing_comp_t{
	const double *pred_val;
	decreasing_comp_t(const double *_val): pred_val(_val) {}
	bool operator()(const size_t i, const size_t j) const {return pred_val[j] < pred_val[i];}
};

void sort_idx_by_val(const double *pred_val, size_t len, size_t *idx, size_t topk) {
	size_t *mid = idx+(topk > len? len : topk);
	std::partial_sort(idx, mid, idx+len, decreasing_comp_t(pred_val));
}

inline double gain(double rel) { return exp2(rel)-1;}
inline double discount(int l) { return 1.0/log2(l+2);}

// input: idx is an sorted index array of length=len
// output: dcg is the array of length=topk with accumuated dcg information
// return: dcg@topk
double compute_dcg(const double *true_rel, size_t *sorted_idx, size_t len, int topk, double *dcg) {
	int levels = topk>len? len : topk;
	double cur_dcg = 0.0;
	for(int l = 0; l < levels; l++) {
		cur_dcg += gain(true_rel[sorted_idx[l]]) * discount(l);
		if(dcg)
			dcg[l] = cur_dcg;
	}
	if(dcg)
		for(int l = levels; l < topk; l++)
			dcg[l] = cur_dcg;
	return cur_dcg;
}

