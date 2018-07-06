#include "math_cuda.h"

void lanczos_facto(Mat *A, float *v0, int k, int m, Mat *Vm, Mat *Tm, float *fm) {
  float b1 = 0;
  float *wPrime = malloc(sizeof(float) * A->m);
  float *v = malloc(sizeof(float) * A->m);
  vect_copy(v0, v, A->m);
  for (int j = k - 1; j < m - 1; j++) {
    // wj' = A * vj
    vect_prod_mat(A, v, wPrime);

    // Vm(:,j) = vj
    vect_mat_copy(Vm, v, j);

    //T(j,j) = wj'.T * vj
    float a1 = vect_dot(wPrime, v, A->m);
    Tm->data[(Tm->m * j) + j]  = a1;
    // wj = wj' - T(j,j)*vj - Bj*vj-1
    vect_scalar(v, a1, A->m);
    if (j > 0) {
      float* vjOld = get_column(Vm, j - 1);
      vect_scalar(vjOld, b1, A->m);
      float *tmp = malloc(sizeof(float) * A->m);
      vect_substract(tmp, wPrime, v, A->m);
      vect_substract(fm, tmp, vjOld, A->m);
      free(tmp);
      free(vjOld);
    } else {
      vect_substract(fm, wPrime, v, A->m);
    }
    // Bj = ||wj||
    b1 = vect_norm(fm, A->m);
    Tm->data[(Tm->m * j) + j + 1] = b1;
    Tm->data[Tm->m * (j + 1) + j] = b1;
    free(v);
    v = vect_divide_by_scalar(fm, b1, A->m);
  }
  vect_mat_copy(Vm, v, m - 1);
  vect_prod_mat(A, v, fm);
  Tm->data[m * m - 1] = vect_dot(fm, v, A->m);
  free(v);
  free(wPrime);
}
float *lanczos_ir(Mat *A, float *v0, int k, int m) {
  Mat *Vm = matrix_zeros(A->n, m);
  Mat *Tm = matrix_zeros(m, m);
  float residual = 1;
  float *fm = malloc(sizeof(float) * A->m);
  Mat *Qm = NULL;
  Mat *QT = NULL;
  Mat *QmT = NULL;
  float eps = 0.00001;
  lanczos_facto(A, v0, 1, m, Vm, Tm, fm);
  int nb_iter = 0;
  while(residual > eps || nb_iter < 100) {
    float *eigs = rritz(Tm, &residual, fm, k, residual);
    Qm = matrix_eye(m, m);
    for (int j = m - k; j >= 0; j--) {
      Mat *mat_tmp = matrix_new(Tm->m, Tm->n);
      Mat *eye = matrix_eye(Tm->n, Tm->n);
      Mat *R = matrix_zeros(Tm->m, Tm->n);
      Mat *Q = matrix_zeros(Tm->m, Tm->n);
      // s * I(:,:)
      matrix_scalar(eye, eigs[j]);
      // H(:,:) - s * I(:,:)
      matrix_sub(Tm, eye, mat_tmp);
      #ifdef NAIVE
        qr_householder(mat_tmp, R, Q);
      #elif INTEL_MKL
        intel_mkl_qr(mat_tmp, R, Q);
      #endif
      // Hm = Qj*HmQj
      QT = matrix_transpose(Q);
      Mat *tmp_mul = matrix_zeros(QT->m, Tm->n);
      matrix_mul_bis(tmp_mul, QT, Tm);
      matrix_mul_bis(Tm, tmp_mul, Q);
      QmT = matrix_transpose(Qm);
      matrix_mul_bis(Qm, QmT, Q);

      // delete all temporaries variables
      matrix_delete(tmp_mul);
      matrix_delete(mat_tmp);
      matrix_delete(eye);
      matrix_delete(R);
      matrix_delete(Q);
      matrix_delete(QT);
      matrix_delete(QmT);
    }
    free(eigs);
    float *tmp_fm = malloc(sizeof(float) * A->m);
    float *tmp_fm2 = malloc(sizeof(float) * A->m);

    vect_copy(fm, tmp_fm, A->m);
    vect_scalar(tmp_fm, Qm->data[m * k], A->m);
    float *QmK = get_column(Qm, k + 1);
    vect_prod_mat(Vm, QmK, tmp_fm2);
    vect_add(fm, tmp_fm2, tmp_fm, A->m);
    float *Vk = get_column(Vm, k);

    // Recreate value Vm, TM, fm
    matrix_delete(Vm);
    matrix_delete(Tm);
    free(fm);

    Vm = matrix_zeros(A->n, m);
    Tm = matrix_zeros(m, m);
    fm = malloc(sizeof(float) * A->m);
    lanczos_facto(A, Vk, k, m, Vm, Tm, fm);

    // delete all temporary variables
    free(QmK);
    matrix_delete(Qm);
    free(Vk);
    free(tmp_fm);
    free(tmp_fm2);

    nb_iter++;
  }

  matrix_delete(Vm);
  Mat *eigVector = matrix_zeros(Tm->m, Tm->n);
  float *eigs = qr_alg_eigen(Tm, eigVector);
  matrix_delete(eigVector);
  matrix_delete(Tm);
  free(fm);
  return eigs;
}
