#include "math.h"
#include <omp.h> // OpenMP
// #include "mkl.h" // Intel MKL
#include <lapacke.h>

Mat *arnoldi_iteration(Mat *A, float *v0, int MAX_ITER) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  float *w = NULL;
  float *fm = NULL;
  float *v = NULL;
  Mat *VmReduce = NULL;
  float *h = NULL;
  for (int j = 0; j < MAX_ITER; j++) {
    if (j == 0) {      
      // w = A * v0
      w = vect_prod_mat(A, v0);
      // Vm(:, j) = v0
      vect_mat_copy(Vm, v0, j);
      // Hm(1,1) = v0.T * w
      Hm->data[0] = vect_dot(v0, w, A->n);
      // fm = w - v0 * v0.T * w
      fm = compute_fm(v0, v0, w, A->n);
    } else {
      // v = fm/||fm||
      v = vect_divide_by_scalar(fm, vect_norm(fm, A->n), A->n);
      // w = A * v
      w = vect_prod_mat(A, v);
      // Vm(:, j) = v
      vect_mat_copy(Vm, v, j);
      // Hm(j, j − 1) = ||fm||
      Hm->data[j * Hm->n + (j-1)] = vect_norm(fm, A->n);
      // Vm(:, 1:j)
      VmReduce = matrix_reduce(Vm, j + 1);
      // h = Vm(:,1 : j).T ∗ w
      h = vect_prod_mat_trans(VmReduce, w);
      // fm = w − Vm(:,1 : j)∗ h
      vect_substract(fm, w, vect_prod_mat(VmReduce, h), A->m);
      // Hm(1 : j, j) = h
      vect_mat_copy_cond(Hm, h, j);

    }
  }
  return Hm;
}

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
    A->data[i] = in[i / 3][i % 3];
  float start = omp_get_wtime();
  eigen_values(A);
  float stop = omp_get_wtime();
  printf("Time : %lf\n", stop-start);
  matrix_delete(A);
  return 0;
}



