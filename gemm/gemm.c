#include "cpu/cpu_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define error(msg) fprintf(stderr, "%s(%u): %s", __FILE__, __LINE__, msg);

void fill_matrix(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {
	uint32_t len = n_rows * n_columns;
	for(uint32_t i = 0; i < len; i++) {
		m[i] = rand();
	}
}

int check(dtype_t* c, dtype_t* correct, uint32_t n_rows, uint32_t n_columns) {
	uint32_t len = n_rows * n_columns;
	for(uint32_t i = 0; i < len; i++) {
		if(c[i] != correct[i]) {
			return 0;
		}
	}
	return 1;
}

void print_matrix(dtype_t* m, uint32_t n_rows, uint32_t n_columns) {
	for(uint32_t i = 0; i < n_rows; i++) {
		for(uint32_t j = 0; j < n_columns; j++) {
			printf("%.2es ", m[i * n_columns + j]);
		}
		printf("\n");
	}
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
	void (*f)(dtype_t*, dtype_t*, dtype_t*, uint32_t, uint32_t, uint32_t);
} EvaluationSuite;

int evaluate(EvaluationSuite suite) {

	// BLOCKED with packing
	memset(suite.C, 0x00, suite.ni * suite.nj * sizeof(dtype_t));

	double start = omp_get_wtime();
	suite.f(suite.C, suite.A, suite.B, suite.ni, suite.nj, suite.nk);
	double stop = omp_get_wtime();

	if(suite.check && !check(suite.C, suite.correct, suite.ni, suite.nj)) {
		printf("C matrix is wrong\n");
		return 1;
	}

	if(!suite.quiet) {
		printf("\t[%s]: %.2es\n", suite.name, stop-start);
	}
	return 0;
}

int main(int argc, char** argv) {
	srand(0);

	#ifdef DEBUG
		uint32_t ni = 3;
		uint32_t nj = 3;
		uint32_t nk = 3;
	#else
		uint32_t ni = 1024;
		uint32_t nj = 1024;
		uint32_t nk = 1024;
	#endif

	if(argc > 1 && sscanf(argv[0], "%u", &ni) != 1) {
		error("invalid ni");
	}
	if(argc > 2 && sscanf(argv[1], "%u", &nj) != 1) {
		error("invalid nj");
	}
	if(argc > 3 && sscanf(argv[1], "%u", &nk) != 1) {
		error("invalid nk");
	}

	dtype_t* correct = malloc(ni * nj * sizeof(dtype_t));
	dtype_t* C = malloc(ni * nj * sizeof(dtype_t));
	dtype_t* A = malloc(ni * nk * sizeof(dtype_t));
	dtype_t* B = malloc(nk * nj * sizeof(dtype_t));

	fill_matrix(A, ni, nk);
	fill_matrix(B, nk, nj);

	memset(correct, 0x00, ni * nj * sizeof(dtype_t));

	EvaluationSuite suite = {
		.ni = ni,
		.nj = nj,
		.nk = nk,
		.correct = correct,
		.C = correct,
		.A = A,
		.B = B,
		.name = "NAIVE",
		.check = 0,
		.f = gemm_rrc_naive
	};

	printf("GEMM\n");
	evaluate(suite);
	suite.C = C;
	suite.correct = correct;
	suite.check = 1;

	suite.f = gemm_rrc_blocked_without_packing;
	suite.name = "BLOCKED";
	evaluate(suite);

	suite.f = gemm_rrc_blocked_with_packing;
	suite.name = "BLOCKED & PACKING";
	evaluate(suite);

	free(correct);
	free(C);
	free(A);
	free(B);
}
