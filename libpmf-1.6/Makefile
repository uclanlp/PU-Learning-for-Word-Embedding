CXX=g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -ffast-math -pipe -g -fPIC
#CXXFLAGS=-fopenmp -march=native -ffast-math -O3 -fPIC
VERSION=1.6

#CXXFLAGS+= -D_USE_FLOAT_

BLASFLAGS=-llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib  #GCC in UTCS
BLASFLAGS=-llapack -lptf77blas -lptcblas -latlas -L/home/01845/rofuyu/.local/lib  # GCC in TACC
BLASFLAGS=-lblas -llapack
LIBSFLAGS=$(BLASFLAGS)

all: omp-pmf-train converter

omp-pmf-train: pmf-train.cpp pmf.o ccd-r1.o als.o sgd.o
	${CXX} ${CXXFLAGS} -o omp-pmf-train pmf-train.cpp pmf.o ccd-r1.o als.o sgd.o ${LIBSFLAGS}

omp-pmf-predict: pmf-predict.cpp pmf.o
	${CXX} ${CXXFLAGS} -o omp-pmf-predict pmf-predict.cpp pmf.o

converter: converter.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -o converter converter.cpp

ccd-r1.o: ccd-r1.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -c -o ccd-r1.o ccd-r1.cpp

als.o: als.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -c -o als.o als.cpp

sgd.o: sgd.cpp sparse_matrix.h block_matrix.h
	${CXX} ${CXXFLAGS} -c -o sgd.o sgd.cpp

pmf.o: pmf.h pmf.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -c -o pmf.o pmf.cpp

smat.o: smat.cpp smat.h
	${CXX} ${CXXFLAGS} -c -o smat.o smat.cpp

tar:
	make clean; cd ../;  tar cvzf libpmf-${VERSION}.tgz libpmf-${VERSION}/
zip:
	make clean; cd ../;  zip -r libpmf-${VERSION}.zip libpmf-${VERSION}/

clean:
	make -C matlab clean
	rm -rf  omp-pmf* converter *.o

