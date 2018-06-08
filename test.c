#include <stdio.h>
#include <lapacke.h>

int main(){
	double matrix[9] = {1,2,3,4,5,6,7,8,9};
	int m = 3;
	int n = 3;
	int info = 0;
	int lda = m;
	int lwork = n;
	double work[3];
	double tau[3];
	//QR FACTORIZATION
	int ite = 0;
	while (ite < 5) {
		dgeqrfp_(&m, &n, matrix, &lda, tau, work, &lwork, &info);
		ite++;
	}
	for (int i = 0; i < m; i++){
		for (int j = 0 ; j < n; j++)
			printf(" %f ", matrix[i + j * m]);
		printf("\n");
	}
}
