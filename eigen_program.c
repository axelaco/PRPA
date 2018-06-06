#include "matrix.h"
#include <omp.h> // OpenMP
// #include "mkl.h" // Intel MKL

double house_helper(double *x, int n) {
  double normX = vect_norm(x, n);
    if (normX == 0)
      x[0] = sqrt(2);
    vect_divide(x, normX, n);
    if (x[0] >= 0) {
      x[0] += 1;
      normX = -normX;
    } else
      x[0] = x[0] - 1;

    double absX = absolute(x[0]);

    vect_divide(x, sqrt(absX), n);

    return normX;
}

Mat *house_apply(Mat *U, Mat *Q) {
  if (!Q)
    return NULL;
  if (!U)
    return NULL;
  double *u = NULL;
  for (int i = Q->n - 1; i >= 0; i--) {
    u = get_column(U, i);
    double *tmp = vect_prod_mat(Q, u);
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

  double *x = NULL;
  double *v = NULL;

  for (int k = 0; k < A->n; k++) {
    x = get_column_start(A, k);
  
    v = malloc(sizeof(double) * (A->n - k - 1));
  
    double normX = house_helper(x, A->n);

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

Mat *arnoldi_iteration(Mat *A, double *v0, int MAX_ITER) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  Mat *Hm = matrix_zeros(A->m, A->m);
  Mat *Vm = matrix_zeros(A->m, A->m);
  double *w = NULL;
  double *fm = malloc(sizeof(double) * A->n);
  double *v = NULL;
  for (int j = 0; j < MAX_ITER; j++) {
    if (j == 0) {
      w = vect_prod_mat(A,v0);
      for (int i = 0; i < A->n; i++) 
        Vm->data[i][j] = v0[i];
      Hm->data[0][0] = vect_dot(v0,w, A->n);
      for (int i = 0; i < A->n; i++) {
        double sum = 0;
        for (int j = 0; j < A->n; j++) {
          sum += v0[i] * v0[j] * w[j];
        }
        fm[i] = w[i] - sum;
      }
    } else {
      v = malloc(sizeof(double) * A->n);
      for (int i = 0; i < A->n; i++) {
        v[i] = fm[i] / vect_norm(fm, A->n);
      }
      w = vect_prod_mat(A,v);
      for (int i = 0; i < A->n; i++) {
        Vm->data[i][j] = v[i];
      }
      Hm->data[j][j-1] = vect_norm(fm, A->n);
      double *h = malloc(sizeof(double) * (j + 1));
      for (int i = 0; i <= j; i++) {
        double sum = 0;
        for (int k = 0; k < A->n; k++)
          sum += Vm->data[k][i] * w[k];
        h[i] = sum;
      }
      for (int i = 0; i < A->n; i++) {
        double sum = 0;
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
double *mapTo1d(Mat *A) {
  double *res = malloc(sizeof(double) * A->n * A->n);
  for(int i=0; i<A->m; i++){
    for(int j=0;j<A->n;j++){
      int position = i*A->n + j;
      res[position] = A->data[i][j];
    }
  }
  return res;
}
void eigen_values(Mat *A) {
  
    int nb_iter = 0;
    double *v = calloc(A->n, sizeof(double));
    for (int i = 0; i < A->n - 1; i++)
      v[i] = 1;
    double vNorm =  vect_norm(v, A->n);
    for (int i = 0; i < A->n; i++) {
        v[i] /= vNorm;
    }
    vect_print(v, 3);
    Mat *Ar = arnoldi_iteration(A, v, 6);
    matrix_print(Ar);
   // double *mat1D = mapTo1d(Ar);
   // vect_print(mat1D, 9);
    double tau;
    //LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A->n, A->m, mat1D, A->n, &tau);
}
void init_matrix(double** v, unsigned int SZ)
{
	double tmp;
	for (int i = 0; i < SZ; i++)
	{
    for (int j = 0; j < SZ; j++) {
		  tmp = (double) (rand()%100);
		  v[i][j] = tmp;
    }
	}	
}

double in[][3] = {
    {2,1,2},
    {4,2,3},  
    {3,2,2}
};

int main(void) {
  //init_matrix(A->data, 10);
  // matrix_print(A);
  int tmp;
  Mat *A = matrix_new(10, 10);
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      tmp = rand()%100;
      A->data[i][j] = (float)tmp/10;
    }
  }
  matrix_print(A);
  double start = omp_get_wtime();
  eigen_values(A);
  double stop = omp_get_wtime();
  printf("Time : %lf\n", stop-start);
  matrix_delete(A);
  return 0;
}



