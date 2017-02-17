#include <cstring>
#include "pmf_matlab.hpp"

static void exit_with_help() {
	mexPrintf(
	"Usage: [ret] = pmf_predict_ranking(testR, W, H, topk, ignored)\n"
	"     testR: a sparse matrix or COO full matrix, use [] to denote no ignore\n"
	"     W: m*k dense matrix\n"
	"     H: n*k dense matrix\n"
	"     ignored: a sparse matrix or COO full matrix, use [] to denote no ignore\n"
	"     ret is a structure with the following fields:\n"
	"       ret.pred_topk is topk*length(row_idx) array with the topk items in each column\n"
	"       ret.pos_rank is nr_pos*4 array with [i j rank nr_pos] in each row\n"
	"       ret.map\n"
	"       ret.auc\n"
	"       ret.hlu\n"
	"       ret.ndcg is a 1*topk array\n"
	);
}

static void fake_answer(mxArray *plhs[]) { plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); }

struct rank_entry_t{
	size_t i,j,rank,nr_pos;
	rank_entry_t(size_t i=0, size_t j=0, size_t rank=0, size_t nr_pos=0):
		i(i), j(j), rank(rank), nr_pos(nr_pos) {}
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(nrhs != 5) {
		exit_with_help();
		fake_answer(plhs);
		return;
	}
	const mxArray *mxtestR = prhs[0], *mxW = prhs[1], *mxH = prhs[2], *mxTopK = prhs[3], *mxIG = prhs[4];
	if(mxGetN(mxW) != mxGetN(mxH)) {
		mexPrintf("Error: size(W,2) != size(H,2)\n");
		fake_answer(plhs);
		return;
	}

	size_t topk = (size_t) *mxGetPr(mxTopK);
	size_t rows = (size_t) mxGetM(mxW), cols = (size_t) mxGetM(mxH), dim = (size_t) mxGetN(mxW);

	smat_t testR;
	mxArray_to_smat(mxtestR, testR, rows, cols);

	const bool model_init = false;
	pmf_model_t model(rows, cols, dim, pmf_model_t::ROWMAJOR, model_init);
	mat_t &W = model.W, &H = model.H;

	mxDense_to_matRow(mxW, W);
	mxDense_to_matRow(mxH, H);

	smat_t ignored;
	// mxArray_to_smat handles both CSC and COO formats
	mxArray_to_smat(mxIG, ignored, rows, cols);
	const int idx_base = 1;

	mxArray *mxpred_topk, *mxndcg, *mxmap, *mxauc, *mxhlu, *mxtrue_pos;
	double *pred_topk = mxGetPr(mxpred_topk=mxCreateDoubleMatrix(topk, rows, mxREAL));
	double *ndcg = mxGetPr(mxndcg=mxCreateDoubleMatrix(1,topk, mxREAL));
	double *map = mxGetPr(mxmap=mxCreateDoubleMatrix(1, 1, mxREAL));
	double *auc = mxGetPr(mxauc=mxCreateDoubleMatrix(1, 1, mxREAL));
	double *hlu = mxGetPr(mxhlu=mxCreateDoubleMatrix(1, 1, mxREAL));

	int nr_threads = omp_get_num_procs();
	//printf("nr_threads %d\n", nr_threads);
	omp_set_num_threads(nr_threads);
	std::vector<info_t> info_set(nr_threads);

	typedef std::vector<rank_entry_t> rank_vec_t;
	std::vector<rank_vec_t> rank_set(nr_threads, rank_vec_t());

	for(int th = 0; th < nr_threads; th++) {
		info_set[th].sorted_idx.reserve(cols);
		info_set[th].true_rel.resize(cols, 0);
		info_set[th].pred_val.resize(cols, 0);

		info_set[th].tmpdcg.resize(topk);
		info_set[th].maxdcg.resize(topk);
		info_set[th].dcg.resize(topk);
		info_set[th].ndcg.resize(topk);
		info_set[th].count.resize(topk+1);
		info_set[th].map = 0;
		info_set[th].auc = 0;
		info_set[th].hlu = 0;
	}

#pragma omp parallel for
	for(long i = 0; i < rows; i++) {
		if(testR.nnz_of_row(i) == 0)
			continue;

		rank_vec_t &rank_vec = rank_set[omp_get_thread_num()];
		info_t &info = info_set[omp_get_thread_num()];
		info.sorted_idx.resize(cols);
		info.pred_val.resize(cols);

		size_t row = i;

		size_t nr_ignore = 0;
		unsigned *ignore_list  = NULL;
		if(ignored.nnz > 0) {
			nr_ignore = ignored.nnz_of_row(row);
			ignore_list = &ignored.col_idx[ignored.row_ptr[row]];
		}

		size_t valid_len = 0; // assigned by pmf_prepare_candidates
		pmf_prepare_candidates(cols, info.pred_val.data(), info.sorted_idx.data(), valid_len, nr_ignore, ignore_list);
		model.predict_row(row, valid_len, info.sorted_idx.data(), info.pred_val.data());

		sort_idx_by_val(info.pred_val.data(), valid_len, info.sorted_idx.data(), valid_len);
		for(int t = 0; t < topk; t++)
			pred_topk[topk*i + t] = t<cols? (double) (info.sorted_idx[t] + idx_base): 0.0;

		for(size_t idx = testR.row_ptr[i]; idx != testR.row_ptr[i+1]; idx++)
			info.true_rel[testR.col_idx[idx]] += testR.val_t[idx];

		// MAP & AUC & HLU
		double localmap = 0;
		double localauc = 0;
		double localhlu = 0, localhlu_max = 0;
		const double neutral_rel = 0, halflife = 5; // paremeters for HLU
		size_t neg_cnt = 0, pos_cnt = 0, violating_pairs = 0;
		for(int j = 0; j < valid_len; j++) {
			size_t col = info.sorted_idx[j];
			if(info.true_rel[col] > 0) {
				// j is the rank of this item
				// pos_cnt is the number of "positive" items ranked before this item
				rank_vec.push_back(rank_entry_t(row, col, j, pos_cnt));
				localhlu += (info.true_rel[col]-neutral_rel)*pow(0.5,j/(halflife-1.0));
				localhlu_max += (info.true_rel[col]-neutral_rel)*pow(0.5,pos_cnt/(halflife-1.0));

				pos_cnt += 1;
				localmap += 100*(double)pos_cnt/(double)(j+1);
				violating_pairs += neg_cnt;
			} else {
				neg_cnt += 1;
			}
		}
		if(pos_cnt > 0)
			localmap /= (double) pos_cnt;
		if(pos_cnt > 0 && neg_cnt > 0)
			localauc = (double)(pos_cnt*neg_cnt-violating_pairs)/(double)(pos_cnt*neg_cnt);
		else
			localauc = 1;
		if(pos_cnt > 0 && localhlu_max > 0)
			localhlu = 100*localhlu/localhlu_max;
		if(valid_len > 0) {
			info.map += localmap;
			info.auc += localauc;
			info.hlu += localhlu;
			info.count[topk] ++;
		}
		// MPR?
		// NDCG
		compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.tmpdcg.data());

		valid_len = testR.row_ptr[i+1] - testR.row_ptr[i];
		if(valid_len) {
			info.sorted_idx.resize(valid_len);
			size_t *sorted_idx = info.sorted_idx.data();
			for(size_t idx = testR.row_ptr[i],t=0; idx != testR.row_ptr[i+1]; idx++,t++)
				sorted_idx[t] = (size_t) testR.col_idx[idx];
			sort_idx_by_val(info.true_rel.data(), valid_len, sorted_idx, topk);
			compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.maxdcg.data());

			for(int k = 0; k < topk; k++) {
				double tmpdcg = info.tmpdcg[k];
				double tmpmaxdcg = info.maxdcg[k];
				if(std::isfinite(tmpdcg) && std::isfinite(tmpmaxdcg) && tmpmaxdcg>0) {
					info.dcg[k] += tmpdcg;
					info.ndcg[k] += 100*tmpdcg/tmpmaxdcg;
					info.count[k] ++;
				}
			}
		}
		for(size_t idx = testR.row_ptr[i]; idx != testR.row_ptr[i+1]; idx++)
			info.true_rel[testR.col_idx[idx]] -= testR.val_t[idx];
	}

	// aggregate results from multiple threads
	size_t nr_total_pos = rank_set[0].size();
	info_t &final_info = info_set[0];
	for(int th = 1; th < nr_threads; th++) {
		nr_total_pos += rank_set[th].size();

		info_t &info = info_set[th];
		for(int k = 0; k < topk; k++) {
			final_info.dcg[k] += info.dcg[k];
			final_info.ndcg[k] += info.ndcg[k];
			final_info.count[k] += info.count[k];
		}
		final_info.map += info.map;
		final_info.auc += info.auc;
		final_info.hlu += info.hlu;
		final_info.count[topk] += info.count[topk];
	}

	if(final_info.count[topk] > 0) {
		final_info.map /= (double) final_info.count[topk];
		final_info.auc /= (double) final_info.count[topk];
		final_info.hlu /= (double) final_info.count[topk];
	}

	*map = final_info.map;
	*auc = final_info.auc;
	*hlu = final_info.hlu;
	for(int k = 0; k < topk; k++) {
		//dcg[k] = final_info.dcg[k] / (double) final_info.count[k];
		ndcg[k] = final_info.ndcg[k] / (double) final_info.count[k];
	}

	static const char *field_names[] = {"pred_topk", "pos_rank", "ndcg", "map", "auc", "hlu"};
	static const int nr_fields = 6;
	mxArray *ret = plhs[0] = mxCreateStructMatrix(1, 1, nr_fields, field_names);
	mxSetField(ret, 0, "pred_topk", mxpred_topk);
	mxSetField(ret, 0, "ndcg", mxndcg);
	mxSetField(ret, 0, "map", mxmap);
	mxSetField(ret, 0, "auc", mxauc);
	mxSetField(ret, 0, "hlu", mxhlu);

	{ // pos_rank
		mxArray *mxpos_rank;
		double *pos_rank = mxGetPr(mxpos_rank = mxCreateDoubleMatrix(nr_total_pos,4,mxREAL));
		mxSetField(ret, 0, "pos_rank", mxpos_rank);
		size_t idx = 0;
		for(int th = 0; th < nr_threads; th++) {
			rank_vec_t &rank_vec = rank_set[th];
			for(size_t s = 0; s < rank_vec.size(); s++) {
				// change everything from 0-based to 1-based
				pos_rank[idx+0*nr_total_pos] = rank_vec[s].i+1;
				pos_rank[idx+1*nr_total_pos] = rank_vec[s].j+1;
				pos_rank[idx+2*nr_total_pos] = rank_vec[s].rank+1;
				pos_rank[idx+3*nr_total_pos] = rank_vec[s].nr_pos+1;
				idx++;
			}
		}
	}
}

