#include <stdio.h>

// Function for printing arrays
__host__ void print_matrix(int *matrix, int N) {
	printf("N = %d\n", N);
	for (int i = 0; i<N; ++i) {
		for(int j = 0; j < N; ++j) {
			printf("%d, ", matrix[i*N + j]);
		}
		printf("\n");
	}
}

// Populates NxN matrix with random values between 0 - 9
__host__ void populate(int* matrix, int N) {
	srand(time(NULL));
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < N; j++) {
			matrix[i*N + j] = rand() % 10; 
		}
	}
}

// Exercise 1: Counts number of even values in NxN matrix
__global__ void count(int *matrix, int *count, int N) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < N*N) {
		if(matrix[id] % 2 == 0) {
			atomicAdd(count, 1);
		}
	}
}

// Exercise 2: Squares NxN matrix
__global__ void square(int *matrix, int *result, int N) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = id / N;
	unsigned j = id % N;
	for (unsigned k = 0; k < N; ++k) {
		result[i*N + j] += matrix[i*N + k] * matrix[k*N + j];
	}
}


int main() {
	// Create matrices of size N
	int N = 16;
	int *h_inmatrix = (int*)malloc(N*N*sizeof(int));
	int *hcount = (int*)malloc(sizeof(int));
	int *d_inmatrix;
	int *d_outmatrix;
	int *dcount;
	cudaMalloc((void **)&d_inmatrix, N*N*sizeof(int));
	cudaMalloc((void **)&d_outmatrix, N*N*sizeof(int));
	cudaMalloc((void **)&dcount, sizeof(int));

	populate(h_inmatrix, N);
	cudaMemcpy(d_inmatrix, h_inmatrix, N*N*sizeof(int), cudaMemcpyHostToDevice);
	print_matrix(h_inmatrix, N);

	// Calculate number of even values
	*hcount = 0;
	cudaMemcpy(dcount, hcount, sizeof(int), cudaMemcpyHostToDevice);
	count<<<N, N>>>(d_inmatrix, dcount, N);
	cudaMemcpy(hcount, dcount, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Number of Even values: %d\n\n", *hcount);

	// Square Matrix 
	square<<<N, N>>>(d_inmatrix, d_outmatrix, N);
	cudaMemcpy(h_inmatrix, d_outmatrix, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	print_matrix(h_inmatrix, N);

	cudaFree(d_inmatrix);
	cudaFree(d_outmatrix);
	cudaFree(dcount);
	free(h_inmatrix);

	return 0;
}
