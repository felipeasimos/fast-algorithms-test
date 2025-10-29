#include "cpu/cpu_gemm.h"
#include "gpu/gpu.h"
#include "gpu/gpu_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define error(msg) fprintf(stderr, "%s(%u): %s", __FILE__, __LINE__, msg);

void fill_matrix(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {
	uint32_t len = n_rows * n_columns;
	for(uint32_t i = 0; i < len; i++) {
		// avoid numeric errors
		m[i] = rand() & 1 ? 0.5 : 1;
	}
}

int check(dtype_t* c, dtype_t* correct, uint32_t n_rows, uint32_t n_columns) {
	uint32_t len = n_rows * n_columns;
	for(uint32_t i = 0; i < len; i++) {
		if(fabs(c[i] - correct[i]) > 0.001) {
			return i;
		}
	}
	return 0;
}

void print_matrix(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			printf("%.2es ", m[i * n_columns + j]);
		}
		printf("\n");
	}
}

void convert_row_major_to_column_major(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {
	dtype_t* m_new = malloc(sizeof(dtype_t) * n_rows * n_columns);
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			m_new[j * n_rows + i] = m[i * n_columns + j]; 
		}
	}
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			uint32_t index = j * n_rows + i;
			m[index] = m_new[index]; 
		}
	}
	free(m_new);
}

void convert_column_major_to_row_major(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {

	dtype_t* m_new = malloc(sizeof(dtype_t) * n_rows * n_columns);
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			m_new[i * n_columns + j] = m[j * n_rows + i]; 
		}
	}
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			uint32_t index = i * n_columns + j;
			m[index] = m_new[index]; 
		}
	}
	free(m_new);
}

typedef struct {
	uint32_t ni;
	uint32_t nj;
	uint32_t nk;
	dtype_t* correct;
	dtype_t* C;
	dtype_t* A;
	dtype_t* B;
	char* name;
	int quiet;
	int check;
	void (*f)(void*, dtype_t*, dtype_t*, dtype_t*, uint32_t, uint32_t, uint32_t);
	GPUData gpu;
} EvaluationSuite;

int evaluate(EvaluationSuite* suite, double* time) {

	memset(suite->C, 0x00, suite->ni * suite->nj * sizeof(dtype_t));

	double start = omp_get_wtime();
	suite->f(&suite->gpu, suite->C, suite->A, suite->B, suite->ni, suite->nj, suite->nk);
	double stop = omp_get_wtime();
	*time = stop - start;

	int error_index = 0;
	if(suite->check && (error_index = check(suite->C, suite->correct, suite->ni, suite->nj))) {
		printf("C matrix is wrong for [%s]: correct[%u] != C[%u] (%f != %f)\n", suite->name, error_index, error_index, suite->correct[error_index], suite->C[error_index]);
		printf("correct:\n");
		print_matrix(suite->correct, suite->ni, suite->nj);
		printf("C:\n");
		print_matrix(suite->C, suite->ni, suite->nj);
		return 1;
	}

	if(!suite->quiet) {
		printf("\t[%s]: %.2es\n", suite->name, *time);
	}
	return 0;
}

EvaluationSuite createSuite(uint32_t ni, uint32_t nj, uint32_t nk, int check) {	
	EvaluationSuite suite = {
		.ni = ni,
		.nj = nj,
		.nk = nk,
		.correct = malloc(ni * nj * sizeof(dtype_t)),
		.C = NULL,
		.A = malloc(ni * nk * sizeof(dtype_t)),
		.B = malloc(nk * nj * sizeof(dtype_t)),
		.name = "NAIVE",
		.check = 0,
		.f = gemm_rrc_naive
	};
	fill_matrix(suite.A, ni, nk);
	fill_matrix(suite.B, nk, nj);

	suite.C = suite.correct;
	memset(suite.correct, 0x00, ni * nj * sizeof(dtype_t));

	if(check) {
		double tmp;
		evaluate(&suite, &tmp);
	}
	suite.check = check;
	suite.C = malloc(ni * nj * sizeof(dtype_t));
	suite.gpu = initGPUData();
	return suite;
}

void freeSuite(EvaluationSuite suite) {
	free(suite.correct);
	free(suite.C);
	free(suite.A);
	free(suite.B);
	freeGPUData(suite.gpu);
}

int createPlotRow(EvaluationSuite suite, FILE* file) {

	double time = 0.0F;
	suite.f = gemm_rrc_blocked_without_packing;
	suite.name = "BLOCKED";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	fprintf(file, "%.2es,", time);

	suite.f = gemm_rrc_blocked;
	suite.name = "BLOCKED & PACKING";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	fprintf(file, "%.2es,", time);

	suite.f = gemm_ccr_blocked_avx;
	suite.name = "BLOCKED & PACKING & AVX (CCR)";
	convert_row_major_to_column_major(suite.correct, suite.ni, suite.nj);
	convert_row_major_to_column_major(suite.A, suite.ni, suite.nk);
	convert_column_major_to_row_major(suite.B, suite.nk, suite.nj);
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	convert_column_major_to_row_major(suite.correct, suite.ni, suite.nj);
	convert_column_major_to_row_major(suite.A, suite.ni, suite.nk);
	convert_row_major_to_column_major(suite.B, suite.nk, suite.nj);
	fprintf(file, "%.2es,", time);

	suite.f = gemm_rrc_to_rrr_blocked_avx;
	suite.name = "BLOCKED & PACKING & AVX (RRC to RRR packing)";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	fprintf(file, "%.2es,", time);

	suite.f = gemm_rrc_blocked_avx;
	suite.name = "BLOCKED & PACKING & AVX (RRC with reduction)";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	fprintf(file, "%.2es,", time);

	suite.f = gemm_rrc_blocked_avx_and_omp;
	suite.name = "BLOCKED & PACKING & AVX (RRC with reduction) & OMP";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	fprintf(file, "%.2es,", time);

	suite.f = (void (*)(void *, dtype_t *, dtype_t *, dtype_t *, uint32_t, uint32_t, uint32_t))gemm_gpu;
	suite.name = "GPU + Copies";
	if(evaluate(&suite, &time)) {
		goto defer;
	}
	printf("\t[GPU] %.2es\n", suite.gpu.time);
	fprintf(file, "%.2es,", suite.gpu.time);
	fprintf(file, "%.2es\n", time);

	return 0;
defer:
	return 1;
}

int createPlot(char* output_path) {
	FILE* f = NULL;
	EvaluationSuite suite = {0};
	if((f = fopen(output_path, "w")) == NULL) {
		error("Error when opening file\n");
		goto defer;
	}
	fprintf(f, "N,BLOCKED,BLOCKED & PACKING,BLOCKED & PACKING & AVX (CCR),BLOCKED & PACKING & AVX (RRC to RRR packing),BLOCKED & PACKING & AVX (RRC with reduction),BLOCKED & PACKING & AVX (RRC with reduction) & OMP,GPU,GPU+Copies");
	#ifdef DEBUG
	int check = 1;
	#else
	int check = 0;
	#endif
	for(uint32_t n = 1; n < 100000; n *= 2) {
		suite = createSuite(n, n, n, check);
		fprintf(f, "%u,", n);
		if(createPlotRow(suite, f)) {
			error("Error when creating row\n");
			goto defer;
		}
		freeSuite(suite);
		fflush(f);
	}
	fclose(f);

	return 0;
defer:
	freeSuite(suite);
	return 1;
}

int main(void) {
	srand(0);

	return createPlot("plot.csv");
}
