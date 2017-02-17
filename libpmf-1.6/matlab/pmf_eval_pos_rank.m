function ret = pmf_eval_pos_rank(pos_rank, topk, ignore, weight)
	if nargin < 2
		disp('ret = pmf_eval_pos_rank(pos_rank, topk [, ignore = sparse(rows,cols), weight = ones(rows, 1))');
		disp('      pos_rank: ret.pos_rank, where ret = pmf_predict_ranking(data.Rt, m.W, m.H, topk, data.R);');
		disp('      topk: topk ndcg');
		disp('      ignore: a sparse matrix to ignore e.g., data.R');
		disp('      weight: weight for weighted evaluation (default: uniform)');
		ret = [];
		return 
	end

	I = pos_rank(:,1);
	J = pos_rank(:,2);
	R = pos_rank(:,3);
	P = pos_rank(:,4);

	if nargin < 3
		ignore = sparse(max(I),max(J));
	end

	if ~issparse(ignore)
		ignore = sparse(ignore(:,1), ignore(:,2), ignore(:,3));
	end

	dim = [max(I), size(ignore, 2)];
	ignore = ignore(1:dim(1), 1:dim(2));

	if nargin < 4
		weight = ones(dim(1),1);
	else 
		weight = weight(1:dim(1));
	end

% MAP
	tmp = sparse(I, J, 100*P./R, dim(1), dim(2));
	map = sum(tmp,2) ./ sum(tmp~=0,2); 
	idx = find(map > 0);
	ret.map = weight(idx)'*map(idx)/sum(weight(idx));

% AUC
	tmp = sparse(I, J, P, dim(1), dim(2));
	pos_cnt = max(tmp,[],2);
	neg_cnt = size(ignore,2) - sum(ignore~=0,2) - pos_cnt; 

	tmp = sparse(I, J, R-P, dim(1), dim(2)); % compute violating pairs
	auc = 1.0 - sum(tmp,2) ./ (pos_cnt .* neg_cnt);
	idx = find(auc > 0);
	ret.auc = weight(idx)'*auc(idx)/sum(weight(idx));

% ndcg
	ret.ndcg = zeros(1,topk);
	maxdcg = zeros(dim(1), 1);
	dcg = zeros(dim(1), 1);
	for k = 1:topk
		idx = find(R == k);
		tmp = sparse(I(idx), J(idx), 1./log2(1+R(idx)), dim(1), dim(2));
		dcg = dcg + sum(tmp,2);
		idx = find(P == k);
		tmp = sparse(I(idx), J(idx), 1./log2(1+P(idx)), dim(1), dim(2));
		maxdcg = maxdcg + sum(tmp,2);
		ndcg = 100*dcg ./ maxdcg; 
		idx = find(maxdcg > 0);
		ret.ndcg(k) = weight(idx)'*ndcg(idx)/sum(weight(idx)); 
	end

% HLU See http://research.microsoft.com/pubs/69656/tr-98-12.pdf
	alpha = 5;
	Ra = (0.5).^((R-1)./(alpha-1));
	RaMax = (0.5).^((P-1)./(alpha-1));
	Ra = sum(sparse(I, J, Ra, dim(1), dim(2)), 2);
	RaMax = sum(sparse(I, J, RaMax, dim(1), dim(2)), 2);
	Ra = Ra./RaMax;
	idx = find(RaMax > 0);
	ret.hlu = 100*weight(idx)'*Ra(idx)/sum(weight(idx));
end
