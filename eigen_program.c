#include "matrix.h"

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


void eigen_values(Mat *A) {
    int nb_iter = 0;
    Mat **QR = NULL;
    while (is_triangle_sup(A) == 0) {
        QR = qr_householder(A);
        A = matrix_mul(QR[1], QR[0]);
        matrix_print(A);
        nb_iter++;
    } 

}
double in[][10] = {
	{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	{ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
	{ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
	{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	{ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
	{ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
};

int main(void) {
  Mat *A = matrix_new(10,10);
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      A->data[i][j] = in[i][j];
  }
  eigen_values(A);
  matrix_delete(A);
  return 0;
}



