#include "math.h"
#include <omp.h> // OpenMP
#include <time.h>
#define N 1000
#define K 100
#define M 200

// QSort Algorithm
static int my_compare (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return abs(*pb) - abs(*pa);
}

void iram(Mat *A, float *v0, int k, int MAX_ITER);
// https://blogs.mathworks.com/cleve/2016/10/03/householder-reflections-and-the-qr-decomposition/
void H(float *u, Mat *R) {
  // u.T*R
  float *res = vect_prod_mat_trans(R, u);
  // u * (u.T * R)
  float *res2 = malloc(sizeof(float) * R->m * R->m);
  for (int i = 0; i < R->m; i++) {
    for (int j = 0; j < R->n; j++) {
      res2[i * R->m + j] = u[i] * res[j];
    }
  }
  free(res);
  // H - u * (u.T * R)
  for (int i = 0; i < R->m * R->m; i++) {
    R->data[i] -= res2[i];
  }
  free(res2);
}

void house_helper(float *x, int n) {
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
}
void house_apply(Mat *U, Mat *Q) {
  if (!Q)
    return;
  if (!U)
    return;
  float *u = NULL;
  for (int j = U->n - 1; j >= 0; j--) {
    u = get_column(U, j);
    H(u, Q);
    free(u);
  }
}

/*
    Param: Matrix to decompose in QR
    Return: [Q, R]
*/

void qr_householder(Mat *A, Mat *R, Mat *Q1) {
  if (!A)
    return;
  if (!R)
    return;
  if (!Q1)
    return;
  matrix_copy(A, R);
  Mat *U = matrix_zeros(A->m, A->n);
  if (!U || !R)
    return;

  float *u = NULL;

  for (int k = 0; k < A->n - 1; k++) {
    u = get_column_start(R, k);
    house_helper(u, A->n - k);
    // U(j:m, j) = u
    vect_mat_copy_cond(U, u, k, 1);
    // R(j:m,j:n)
    Mat *Rreduce =  matrix_reduce_cond(R, k);
    // H(u, R(j:m, j:n))
    H(u, Rreduce);
    // R(j:m, j:n) = H(u, R(j:m, j:n))
    matrix_copy_cond(R, Rreduce, k);
    // R(j+1:m, j) = 0
    for (int j = (k + 1) * R->m + k; j < R->m * R->n; j+=R->m) {
      R->data[j] = 0;
    }
    matrix_delete(Rreduce);
    free(u);
  }
  Mat *Q = matrix_eye(U->m, U->n);
  matrix_copy(Q, Q1);
  house_apply(U,  Q1);
  matrix_delete(U);
  matrix_delete(Q);
}

#ifdef INTEL_MKL

void intel_mkl_qr(Mat *Ar, Mat *R, Mat *Q) {
  if (!Ar || !R || !Q)
    return;
  float tau[Ar->m];
  // Compute QR
  LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->data, Ar->m, tau);

  matrix_copy(Ar, Q);
  // Retrieve R tri upper
  LAPACKE_slacpy(LAPACK_ROW_MAJOR, 'U', Ar->m, Ar->n, Ar->data, Ar->m, R->data, Ar->n);
  // Retrieve Q
  LAPACKE_sorgqr(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->m, Q->data, Ar->m, tau);
}
#endif
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
void arnoldi_iteration(Mat *A, float *v0, int k, int MAX_ITER, Mat *Hm, Mat *Vm, float *fm) {
  vect_divide(v0, vect_norm(v0, A->n), A->n);
  /*Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  */
  float *w = malloc(sizeof(float) * A->m);
  //float *fm = NULL;
  float *v = NULL;
  float *VmReduce2 = NULL;
  float *h = NULL;
  float *res = NULL;
  for (int j = k - 1; j < MAX_ITER; j++) {
      //printf("%d Ite\n", j);
      if (j == 0) {
        // w = A * v0
        vect_prod_mat(A, v0, w);
        // Vm(:, j) = v0
        vect_mat_copy(Vm, v0, j);
        // Hm(1,1) = v0.T * w
        Hm->data[0] = vect_dot(v0, w, A->n);

        // fm = w - v0 * v0.T * w
        compute_fm(fm, v0, w, A->n, A->m);
      } else {
        // v = fm/||fm||
        v = vect_divide_by_scalar(fm, vect_norm(fm, A->n), A->n);
        // w = A * v
        vect_prod_mat(A, v, w);
        // Vm(:, j) = v
        vect_mat_copy(Vm, v, j);

        // Hm(j, j − 1) = ||fm||
        Hm->data[j * Hm->n + (j-1)] = vect_norm(fm, A->n);

        Mat *VmReduce = matrix_reduce(Vm, j + 1);
        // h = Vm(:,1 : j).T ∗ w

        float *h = vect_prod_mat_trans(VmReduce, w);;
        // Vm(:, 1:j) * h
        float *tmp = malloc(sizeof(float) * VmReduce->m);
        vect_prod_mat(VmReduce, h, tmp);

        // fm = w − Vm(:,1 : j)∗ h
        vect_substract(fm, w, tmp, A->m);

        // Hm(1 : j, j) = h
        vect_mat_copy_cond(Hm, h, j, 0);

        free(tmp);
        free(v);
        free(h);
        matrix_delete(VmReduce);
      }
    }
    free(w);
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
  //  vect_print(v, A->n);
    float *res = lanczos_ir(A, v, K, M);
    printf("##### Eigen Values: #####\n");
    qsort(res, M, sizeof(*res), my_compare);
    for (int i = 0; i < K; i++)
      printf("%8.5f\n", res[i]);
    free(res);
    free(v);
}

void iram(Mat *A, float *v0, int k, int m) {
  Mat *Hm = matrix_zeros(m, m);
  Mat *Vm = matrix_zeros(A->n, m);
  float *fm = malloc(sizeof(float) * A->m);
  arnoldi_iteration(A, v0, 1, m, Hm, Vm, fm);
  int nb_iter = 0;
  while (nb_iter < 1) {
    printf("Iteration %d\n", nb_iter);
    float *eigenValues = qr_alg_eigen(Hm, NULL);
    Mat *Qm = matrix_eye(Hm->m, Hm->n);
    vect_print(eigenValues, Hm->m);
    for (int j = m - k; j >= 0; j--) {
      Mat *mat_tmp = matrix_new(Hm->m, Hm->n);
      Mat *eye = matrix_eye(Hm->n, Hm->n);
      Mat *R = matrix_zeros(Hm->m, Hm->n);
      Mat *Q = matrix_zeros(Hm->m, Hm->n);
      // s * I(:,:)
      matrix_scalar(eye, eigenValues[j]);
      // H(:,:) - s * I(:,:)
      matrix_sub(Hm, eye, mat_tmp);
      #ifdef NAIVE
        qr_householder(mat_tmp, R, Q);
      #elif INTEL_MKL
        intel_mkl_qr(mat_tmp, R, Q);
      #endif
      // Hm = Qj*HmQj
      Mat *QT = matrix_transpose(Q);
      Hm = matrix_mul(matrix_mul(QT, Hm), Q);
      Mat *QmT = matrix_transpose(Qm);
      Qm = matrix_mul(QmT, Q);
      puts("Hm:");
      matrix_print(Hm);
      puts("Vm:");
      matrix_print(Vm);
    }
    // Vm(:,:) * Qm(:, k + 1)
    float *fm_tmp = malloc(sizeof(float) * A->m);
    if (!fm_tmp)
      return;
    vect_prod_mat(Vm, get_column(Qm, k + 1), fm_tmp);

    vect_scalar(fm_tmp, Hm->data[(k+1) * k], A->m);
    vect_scalar(fm, Qm->data[m * k], A->m);

    vect_add(fm, fm_tmp, fm, A->m);
    Mat *QmReduce = matrix_reduce(Qm, k);

    // Vm(::) * Qm(:,1:k)
    Mat *tmp_2 = matrix_mul(Vm, QmReduce);

    // Vm(:, 1:k) = Vm(::) * QM(:,1:k)
    matrix_copy_sub(tmp_2, Vm, k);
    float *Vk = get_column(Vm, k);
    arnoldi_iteration(A, Vk, k, m, Hm, Vm, fm);
    nb_iter++;
  }
}
float in[][3] = {
    {1,2,3},
    {2,2,4},
    {3,4,3}
};
float in2[][6] = {
    {35,1, 6, 26,19,24},
    {3, 32, 7,21,23,25},
    {31,9,2,22,27,20},
    {8,28,33,17,10,15},
    {30,5,34,12,14,16},
    {4,36,29,13,18,11},
};
void init_random_matrix(Mat *A) {
  float tmp;
  srand(10);
  for (int i = 0; i < (A->m * A->n ); i++) {
    tmp = (float) (rand() % 100);
    A->data[i] = tmp;
  }
}
void init_random_matrix_sym(Mat *A) {
  float tmp;
  srand(10);
  for (int i = 0; i < A->m; i++) {
    for (int j = 0; j < A->n; j++) {
      tmp = (float) (rand() % 100);
      A->data[A->m * i + j] = tmp;
      A->data[A->n * j + i] = tmp;
    }
  }
}
 float in3[] = {
   111.0000305175781250,
 -64.8720779418945312,
 -64.8720779418945312,
 -27.0000362396240234,
 -22.2301902770996094,
 -22.2301902770996094,
 -9.7979869842529297,
 -0.0000078366610978,
 9.7973051071166992,
 27.0011253356933594,
 5.0293035507202148,
 5.0293035507202148,
 49.0527648925781250,
 49.0527648925781250,
 45.0782318115234375,
 45.0782318115234375,
 36.8878211975097656,
 36.8878211975097656,
 23.9167861938476562,
 23.9167861938476562,
 };
int main(void) {

  int tmp;
  Mat *A = matrix_new(N, N);
  /*for (int i = 0; i < N * N; i++)
    A->data[i] = in[i / N][i % N];
  matrix_print(A);
  */
  init_random_matrix_sym(A);
  //matrix_print(A);
  puts("");

  clock_t t = clock();
  eigen_values(A);
  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC;
  printf("Compute EigenValues took %.4f seconds to execute \n", time_taken);
  matrix_delete(A);

  return 0;
}
