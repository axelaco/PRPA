#include "matrix.h"
#include <omp.h> // OpenMP
// #include "mkl.h" // Intel MKL
#include <lapacke.h>

float house_helper(float *x, int n) {
  float normX = vect_norm(x, n);
    if (normX == 0)
      x[0] = sqrt(2);
    vect_divide(x, normX, n);
    if (x[0] >= 0) {
      x[0] += 1;
      normX = -normX;
    } else
      x[0] = x[0] - 1;

    float absX = absolute(x[0]);

    vect_divide(x, sqrt(absX), n);

    return normX;
}

Mat *house_apply(Mat *U, Mat *Q) {
  if (!Q)
    return NULL;
  if (!U)
    return NULL;
  float *u = NULL;
  for (int i = Q->n - 1; i >= 0; i--) {
    u = get_column(U, i);
    float *tmp = vect_prod_mat(Q, u);
    for (int k = 0; k < Q->n; k++) {
      for (int j = 0; j < Q->n; j++) {
        Q->data[j][k] = Q->data[j][k] - (tmp[k] * u[j]);
      }
    }
  }  
  return Q;
}


/*
    Param: Matrix to decompose in QR
    Return: [Q, R]
*/

Mat **qr_householder(Mat *A) {
  if (!A)
    return NULL;
  Mat **res = malloc(sizeof(Mat *) * 2);
  if (!res)
    return NULL;
  for (int i = 0; i < 2; i++) {
    res[i] = malloc(sizeof(Mat));
  }

  Mat *R = matrix_zeros(A->m, A->n);
  Mat *U = matrix_zeros(A->m, A->n);

  if (!U || !R)
    return NULL;

  float *x = NULL;
  float *v = NULL;

  for (int k = 0; k < A->n; k++) {
    x = get_column_start(A, k);
  
    v = malloc(sizeof(float) * (A->n - k - 1));
  
    float normX = house_helper(x, A->n);

    R->data[k][k] = normX; 
  
    vect_mat_copy(U, x, k);  
    
    for (int j = k + 1; j < A->n; j++) {
      v[j - (k + 1)] = 0;
      for (int i = k; i < A->n; i++) {
        v[j - (k + 1)] += U->data[i][k] * A->data[i][j];
      }
    }
    
    for (int j = k + 1; j < A->n; j++) {
      for (int i = k; i < A->n; i++) {
        A->data[i][j] = A->data[i][j] - U->data[i][k] * v[(j - (k + 1))];
      }
    }    
  
    for (int i = k + 1; i < A->n; i++)
      R->data[k][i] = A->data[k][i];
  }

  Mat *Q = house_apply(U,  matrix_eye(A->m, A->n));
  res[0] = Q;
  res[1] = R;
  return res;
}
int is_triangle_sup(Mat *X){
  for (int i = 1; i < X->m; i++)
    for (int j = 0; j < i; j++)
      if (fabs(X->data[i][j]) > 0.0005)
        return 0;
  return 1;
}
 

Mat *arnoldi_iteration(Mat *A, float *v0, int MAX_ITER) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  float *w = NULL;
  float *fm = malloc(sizeof(float) * A->n);
  float *v = NULL;
  for (int j = 0; j < MAX_ITER; j++) {
    if (j == 0) {
      w = vect_prod_mat(A,v0);
      for (int i = 0; i < A->n; i++) 
        Vm->data[i][j] = v0[i];
      Hm->data[0][0] = vect_dot(v0,w, A->n);
      for (int i = 0; i < A->n; i++) {
        float sum = 0;
        for (int j = 0; j < A->n; j++) {
          sum += v0[i] * v0[j] * w[j];
        }
        fm[i] = w[i] - sum;
      }
    } else {
      v = malloc(sizeof(float) * A->n);
      for (int i = 0; i < A->n; i++) {
        v[i] = fm[i] / vect_norm(fm, A->n);
      }
      w = vect_prod_mat(A,v);
      for (int i = 0; i < A->n; i++) {
        Vm->data[i][j] = v[i];
      }
      Hm->data[j][j-1] = vect_norm(fm, A->n);
      float *h = malloc(sizeof(float) * (j + 1));
      for (int i = 0; i <= j; i++) {
        float sum = 0;
        for (int k = 0; k < A->n; k++)
          sum += Vm->data[k][i] * w[k];
        h[i] = sum;
      }
      for (int i = 0; i < A->n; i++) {
        float sum = 0;
        for (int k = 0; k <= j; k++)
          sum += Vm->data[i][k] * h[k];
        fm[i] = w[i] - sum;
      }
      for (int i = 0; i <= j; i++) {
        Hm->data[i][j] = h[i];
      }
    }
  }
  return Hm;
}
/*
float *mapTo1d(Mat *A) {
  float *res = malloc(sizeof(float) * A->n * A->n);
  for(int i=0; i<A->m; i++){
    for(int j=0;j<A->n;j++){
      int position = i*A->n + j;
      res[position] = A->data[i][j];
    }
  }
  return res;
}
void my_qr(Mat *A) {
	int info = 0;
	int lda = A->m;
	int lwork = A->n;
	float work[3];
	float tau[3];
  float *mat1D = mapTo1d(A);
	//QR FACTORIZATION
	sgeqrfp_(&A->m, &A->n, mat1D, &lda, tau, work, &lwork, &info);
  for (int i = 0; i < lda * lwork; i++) {
    if (i % 3 == 0)
    {
      printf("\n");
    }
    printf("%6.4f ", mat1D[i]);
  }
}
*/
void eigen_values(Mat *A) {
  
    int nb_iter = 0;
    float *v = calloc(A->n, sizeof(float));
    for (int i = 0; i < A->n - 1; i++)
      v[i] = 1;
    float vNorm =  vect_norm(v, A->n);
    for (int i = 0; i < A->n; i++) {
        v[i] /= vNorm;
    }
    vect_print(v, 3);
    Mat *Ar = arnoldi_iteration(A, v, 3);
    matrix_print(Ar);
    //LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A->n, A->m, mat1D, A->n, &tau);
}
void init_matrix(float** v, unsigned int SZ)
{
	float tmp;
	for (int i = 0; i < SZ; i++)
	{
    for (int j = 0; j < SZ; j++) {
		  tmp = (float) (rand()%100);
		  v[i][j] = tmp;
    }
	}	
}

float in[][3] = {
    {2,1,2},
    {4,2,3},  
    {3,2,2}
};

int main(void) {
  int tmp;
  Mat *A = matrix_new(3, 3);
  for (int i = 0; i < 3 * 3; i++)
    A->data[i/3][i%3] = in[i / 3][i % 3];
  float start = omp_get_wtime();
  eigen_values(A);
  float stop = omp_get_wtime();
  printf("Time : %lf\n", stop-start);
  matrix_delete(A);
  return 0;
}



