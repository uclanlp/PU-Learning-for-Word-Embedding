#include <cstring>

#ifdef WITH_SMAT_H
#include "smat.h"
#else
#include "sparse_matrix.h"
#ifdef _USE_FLOAT_
#define val_type float
#else 
#define val_type double
#endif
typedef sparse_matrix<val_type> smat_t;
#endif // end of ifdef WITH_SMAT_H


struct converter_parameter_t{
	size_t cache_size;
	converter_parameter_t(size_t cache_size_=0){ cache_size = cache_size_; }
};

void exit_with_help() {
	printf(
	"Usage: converter [options] data_dir\n"
	"options:\n"
	"    -m cache_size : an integer of the cache_size in MB\n"
	"        0 -- unlimited cache_size (default)\n"
	);
	exit(1);
}

converter_parameter_t parse_command_line(int argc, char **argv) {
	converter_parameter_t param;   // default values have been set by the constructor 
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'm':
				param.cache_size = (size_t) strtol(argv[i], NULL, 10);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if(i >= argc)
		exit_with_help();
	return param;
}

static void filecpy(FILE *dest_fp, FILE *src_fp, size_t n) {
	const size_t chunksize = 10240;
	char buf[chunksize];
	for(size_t i = 0; i < n; i += chunksize) {
		size_t buf_len = std::min(n - i, chunksize); 
		if(fread(&buf[0], sizeof(char), buf_len, src_fp) != buf_len)
			fprintf(stderr, "Error: something wrong in filecpy!!\n");
		if(fwrite(&buf[0], sizeof(char), buf_len, dest_fp) != buf_len)
			fprintf(stderr, "Error: something wrong in filecpy!!\n");
	}
}

static size_t get_filesize(const char *input) {
	FILE *fp = fopen(input, "rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: can't read %s!!\n", input);
		return 0;
	}
	fseek(fp, 0, SEEK_END);
	size_t filesize = ftell(fp);
	fclose(fp);
	return filesize;
}

static size_t get_linecount(const char *input, long start_pos=0, long end_pos=-1) {
	FILE *fp = fopen(input, "rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: can't read %s!!\n", input);
		return 0;
	}
	const int chunksize = 10240;
	char buf[chunksize];
	size_t filelen;
	if(end_pos == -1) {
		fseek(fp, 0, SEEK_END);
		end_pos = ftell(fp) - start_pos;
	}
	size_t linecount = 0;
	fseek(fp, start_pos, SEEK_SET);
	for(size_t cnt = start_pos; cnt < end_pos; cnt += chunksize) {
		size_t buf_len = std::min((size_t)chunksize, end_pos-cnt);
		size_t tmp;
		if(buf_len != (tmp=fread(&buf[0], sizeof(char), buf_len, fp))) {
			fprintf(stderr, "Error: something wrong in linecount() expect %ld bytes but read %ld instead!!\n", tmp, buf_len);
		}
		for(size_t i = 0; i < buf_len; i++)
			if(buf[i] == '\n') linecount++;
	}
	fclose(fp);
	return linecount;
}

static void text_to_PETSc(size_t rows, size_t cols, size_t nnz, 
		const char *src_filename, const char *dest_filename, size_t start_pos=0) {
	file_iterator_t<val_type> entry_it(nnz, src_filename, start_pos);
	smat_t R;
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	R.save_PETSc_to_file(dest_filename);
}

static void PETSc_merger(FILE *dest_fp, std::vector<FILE*> src_fp_set, size_t rows, size_t cols) {
		// Same as save_PETSc_to_file() in sparese_matrix.h
		const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
		int32_t int_buf[3] = {(int32_t)LONG_FILE, (int32_t)rows, (int32_t)cols};
		size_t headersize = 3*sizeof(int32_t);

		size_t num_blocks = src_fp_set.size();
		size_t nnz = 0;
		std::vector<size_t> block_nnz(num_blocks);
		for(size_t i = 0; i < num_blocks; i++) {
			fseek(src_fp_set[i], headersize, SEEK_SET);
			if(fread(&block_nnz[i], sizeof(size_t), 1, src_fp_set[i]) != 1)
				fprintf(stderr, "Error: something wrong in limited_memory_converter()!!\n");
			nnz += block_nnz[i];
		}
		fwrite(&int_buf[0], sizeof(int32_t), 3, dest_fp);
		fwrite(&nnz, sizeof(size_t), 1, dest_fp);

		// nnz_row
		std::vector<int>nnz_row(rows);
		std::vector<std::vector<int> > nnz_row_set(num_blocks, std::vector<int>(rows));
		for(size_t i = 0; i < num_blocks; i++) {
			if(fread(&nnz_row_set[i][0], sizeof(int), rows, src_fp_set[i]) != rows)
				fprintf(stderr, "Error: something wrong in limited_memory_converter()!!\n");
			for(size_t r = 0; r < rows; r++)
				nnz_row[r] += nnz_row_set[i][r];
		}
		fwrite(&nnz_row[0], sizeof(int), rows, dest_fp);
		// col_idx
		for(size_t r = 0; r < rows; r++) {
			for(size_t i = 0; i < num_blocks; i++)
				filecpy(dest_fp, src_fp_set[i], sizeof(unsigned)*nnz_row_set[i][r]);
		}
		// val
		for(size_t r = 0; r < rows; r++) {
			for(size_t i = 0; i < num_blocks; i++)
				filecpy(dest_fp, src_fp_set[i], sizeof(val_type)*nnz_row_set[i][r]);
		}
}

static void limited_memory_converter(size_t rows, size_t cols, size_t nnz,
		const char *src_filename, const char *dest_filename, size_t cache_size = 0) {
	if(cache_size == 0) { // no limit on memory usage
		text_to_PETSc(rows, cols, nnz, src_filename, dest_filename);
	} else {
		// Compute the num of blocks
		size_t filesize = get_filesize(src_filename);
		cache_size = std::max(cache_size >> 1, 1UL) * (1L << 20);
		size_t num_blocks = filesize/cache_size + ((filesize%cache_size!=0)?1:0);

		// Generate offset for each block
		std::vector<size_t> block_offset(num_blocks+1);
		block_offset[0] = 0; block_offset[num_blocks] = filesize;
		FILE *src_fp = fopen(src_filename, "rb");
		for(size_t i = 1; i < num_blocks; i++) {
			block_offset[i] = block_offset[i-1] + cache_size;
			fseek(src_fp, block_offset[i]-1, SEEK_SET);
			while( fgetc(src_fp) != '\n');
			block_offset[i] = ftell(src_fp);
		}
		fclose(src_fp);

		std::vector<size_t> block_nnz(num_blocks);
		std::vector<FILE*> src_fp_set(num_blocks);
		char tmp_filename[1024];
		size_t total_nnz = 0;
		for(size_t i = 0; i < num_blocks; i++) {
			block_nnz[i] = get_linecount(src_filename, block_offset[i], block_offset[i+1]);
			//printf("bid %ld: offset %ld - %ld nnz %ld\n", i, block_offset[i], block_offset[i+1], block_nnz[i]);
			total_nnz += block_nnz[i];
			sprintf(tmp_filename, "%s-%ld", dest_filename, i);
			text_to_PETSc(rows, cols, block_nnz[i], src_filename, tmp_filename, block_offset[i]);
			src_fp_set[i] = fopen(tmp_filename, "rb");
		}
		FILE *dest_fp = fopen(dest_filename, "wb");
		PETSc_merger(dest_fp, src_fp_set, rows, cols);
		for(size_t i = 0; i < num_blocks; i++) 
			fclose(src_fp_set[i]);
		fclose(dest_fp);
	}
}

int main(int argc, char* argv[]){
	size_t m, n, nnz;
	char filename[1024];
	char src_filename[1024], dest_filename[1024];
	char *srcdir = argv[argc-1];
	converter_parameter_t param = parse_command_line(argc, argv);

	FILE *fp; 
	sprintf(filename, "%s/meta", srcdir);
	fp = fopen(filename, "r");
	if(fscanf(fp, "%lu %lu", &m, &n) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
		return 1;
	}

	if(fscanf(fp, "%lu %s", &nnz, filename) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
		return 1;
	}
	size_t start_pos = 0;
	sprintf(src_filename, "%s/%s", srcdir, filename);
	sprintf(dest_filename, "%s/%s.petsc", srcdir, filename);
	printf("Converting training set %s (m %ld n %ld nnz %ld)\n", filename, m, n, nnz);
	limited_memory_converter(m, n, nnz, src_filename, dest_filename, param.cache_size);

	if(fscanf(fp, "%lu %s", &nnz, filename) != EOF){
		sprintf(src_filename, "%s/%s", srcdir, filename);
		sprintf(dest_filename, "%s/%s.petsc", srcdir, filename);
		printf("Converting test set %s (m %ld n %ld nnz %ld)\n", filename, m, n, nnz);
		limited_memory_converter(m, n, nnz, src_filename, dest_filename, param.cache_size);
	}

	return 0;
}

