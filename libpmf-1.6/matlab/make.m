function make()
% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
Type = ver;
% This part is for OCTAVE
if(strcmp(Type(1).Name, 'Octave') == 1)
	blas_options ='-L/u/rofuyu/.local/lib -llapack_atlas -lf77blas -lcblas -latlas -lgfortran';
	blas_options ='-lblas -llapack';
	setenv('CXXFLAGS', '-fopenmp -ffast-math -pipe -O3 -DNDEBUG ');
	% pmf_train
	mex('-v', '-lgomp', blas_options, 'pmf_train.cpp', '../pmf.cpp', '../ccd-r1.cpp', '../als.cpp', '../sgd.cpp');
	% pmf_predict
	mex('-v', '-lgomp', 'pmf_predict.cpp', '../pmf.cpp');
	% pmf_predict_topk
	mex('-v', '-lgomp', 'pmf_predict_topk.cpp', '../pmf.cpp');
	% pmf_eval
	mex('-v', '-lgomp', 'pmf_eval.cpp', '../pmf.cpp');
	% pmf_predict_ranking
	mex('-v', '-lgomp', 'pmf_predict_ranking.cpp', '../pmf.cpp');

% This part is for MATLAB
% remove -largeArrayDims on 32-bit machines of MATLAB
else
	verbose = '';
	largearray = '-largeArrayDims';

	if ispc() == 1,  % windows
		compflags = 'COMPFLAGS="$COMPFLAGS /openmp "';
		ldflags = 'LDFLAGS="$LDFLAGS /openmp "';
		cflags = 'CFLAGS="$CFLAGS /openmp "';
		cxxflags = 'CXXFLAGS="$CXXFLAGS /openmp "';

		coptim = 'OPTIMFLAGS="$OPTIMFLAGS /O2 /fp:fast /D_USE_FLOAT_"';
		cxxoptim = '';
		linkoptim = 'LDOPTIMFLAGS="$LDOPTIMFLAGS"';

		blaslib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft','libmwblas.lib');
		lapacklib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft','libmwlapack.lib');
	else
		compflags = 'COMPFLAGS="$COMPFLAGS -fopenmp "';
		ldflags = 'LDFLAGS="$LDFLAGS -fopenmp "';
		cflags = 'CFLAGS="$CFLAGS -fopenmp -I../ "';
		cxxflags = 'CXXFLAGS="$CXXFLAGS -fopenmp -I../ "';

		coptim = 'COPTIMFLAGS="$COPTIMFLAGS -O3 -DNDEBUG -march=native -ffast-math -D _USE_FLOAT_"';
		cxxoptim = 'CXXOPTIMFLAGS="$CXXOPTIMFLAGS -O3 -DNDEBUG -march=native -ffast-math -D _USE_FLOAT_"';
		linkoptim = 'LDOPTIMFLAGS="$LDOPTIMFLAGS -O3"';

		blaslib = '-lmwblas';
		lapacklib = '-lmwlapack';
	end
	% pmf_train
	mex(verbose, largearray, blaslib, lapacklib, compflags, cxxflags, cflags, ldflags, coptim, cxxoptim, linkoptim, ...
			'-cxx', 'pmf_train.cpp','../pmf.cpp', '../ccd-r1.cpp', '../als.cpp', '../sgd.cpp');
	% pmf_predict
	mex(verbose, largearray, blaslib, lapacklib, compflags, cxxflags, cflags, ldflags, ...
			'-cxx', 'pmf_predict.cpp', '../pmf.cpp');
	% pmf_predict_topk
	mex(verbose, largearray, blaslib, lapacklib, compflags, cxxflags, cflags, ldflags, ...
			'-cxx', 'pmf_predict_topk.cpp', '../pmf.cpp');
	% pmf_eval
	mex(verbose, largearray, blaslib, lapacklib, compflags, cxxflags, cflags, ldflags, ...
			'-cxx', 'pmf_eval.cpp', '../pmf.cpp');
	% pmf_predict_ranking
	mex(verbose, largearray, blaslib, lapacklib, compflags, cxxflags, cflags, ldflags, ...
			'-cxx', 'pmf_predict_ranking.cpp', '../pmf.cpp');
end

