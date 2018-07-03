#include "math_complex.h"
#include <omp.h> // OpenMP
#define N 6
#define K 2
//#include <lapacke.h>

// QSort Algorithm
static int my_compare (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   float const *pa = a;
   float const *pb = b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return *pa - *pb;
}


void iram(Mat *A, complex *v0, int k, int MAX_ITER);
// https://blogs.mathworks.com/cleve/2016/10/03/householder-reflections-and-the-qr-decomposition/
void H(complex *u, Mat *R) {
  // u.T*R
  complex *res = vect_prod_mat_trans(R, u);
  // u * (u.T * R)
  complex *res2 = malloc(sizeof(complex) * R->m * R->m);
  for (int i = 0; i < R->m; i++) {
    for (int j = 0; j < R->n; j++) {
      res2[i * R->m + j] = complex_prod(u[i], res[j]);
    }
  }
  free(res);
  // H - u * (u.T * R)
  for (int i = 0; i < R->m * R->m; i++) {
    R->data[i].real -= res2[i].real;
    R->data[i].imag -= res2[i].imag;
  }
  free(res2);
}

void house_helper(complex *x, int n) {
  float normX = vect_norm(x, n);
  complex normCp = (complex){normX, 0};
    if (normX == 0) {
      x[0].real = sqrt(2);
      x[0].imag = 0;
    }
    vect_divide(x, normCp, n);
    if (x[0].real >= 0) { 
      x[0].real += 1;
      normX = -normX;
    } else
      x[0].real = x[0].real - 1;

    float absX = absolute(x[0]);
    absX = sqrt(absX);
    vect_divide(x, (complex){absX, 0}, n);
}
void house_apply(Mat *U, Mat *Q) {
  if (!Q)
    return;
  if (!U)
    return;
  for (int j = U->n - 1; j >= 0; j--) {
    complex *u = get_column(U, j);
    H(u, Q);
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

  complex *u = NULL;

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
      R->data[j].imag = 0;
      R->data[j].real = 0;
    }
  }
  Mat *Q = matrix_eye(U->m, U->n);
  matrix_copy(Q, Q1);
  house_apply(U,  Q1);
}

#ifdef INTEL_MKL

void intel_mkl_qr(Mat *Ar, Mat *R, Mat *Q) {
  if (!Ar || !R || !Q)
    return;
  complex tau[Ar->m];
  // Compute QR
  LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->data, Ar->m, tau);

  matrix_copy(Ar, Q);
  // Retrieve R tri upper
  LAPACKE_clacpy(LAPACK_ROW_MAJOR, 'U', Ar->m, Ar->n, Ar->data, Ar->m, R->data, Ar->n);  
  // Retrieve Q 
 // LAPACKE_sorgqr(LAPACK_ROW_MAJOR, Ar->m, Ar->n, Ar->m, Q->data, Ar->m, tau);
}
#endif

void arnoldi_iteration(Mat *A, complex *v0, int k, int MAX_ITER, Mat *Hm, Mat *Vm, complex *fm) {
  vect_divide(v0, (complex){vect_norm(v0, A->n), 0}, A->n);
  /*Mat *Hm = matrix_zeros(MAX_ITER, MAX_ITER);
  Mat *Vm = matrix_zeros(A->n, MAX_ITER);
  */
  complex *w = malloc(sizeof(complex) * A->m);
  //float *fm = NULL;
  complex *v = NULL;
  complex *VmReduce2 = NULL;
  complex *h = NULL;
  complex *res = NULL;
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
        v = vect_divide_by_scalar(fm, (complex){vect_norm(fm, A->n), 0}, A->n);
        // w = A * v
        vect_prod_mat(A, v, w);
        // Vm(:, j) = v
        vect_mat_copy(Vm, v, j);

        // Hm(j, j − 1) = ||fm||
        Hm->data[j * Hm->n + (j-1)].real = vect_norm(fm, A->n);
        
        Mat *VmReduce = matrix_reduce(Vm, j + 1);
        // h = Vm(:,1 : j).T ∗ w

        complex *h = vect_prod_mat_trans(VmReduce, w);;
        // Vm(:, 1:j) * h
        complex *tmp = malloc(sizeof(complex) * VmReduce->m);
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
void rritz(Mat *Hm, float *f, int k, int selec, double nrmfr) {

}
complex *qr_alg_eigen(Mat *A) {/*
    matrix_print(A);
    Mat *mat_tmp = matrix_new(A->m, A->n);
    Mat *A0 = matrix_new(A->m, A->n);
    matrix_copy(A, A0);
    int k = 0;
    Mat *R = matrix_zeros(A->m, A->n);
    Mat *Q = matrix_zeros(A->m, A->n);
    Mat *Q1 = matrix_eye(A->m, A->n);
    while (k < 50000) {
      // s = A(n,n)
      float s = A->data[A->n * A->m - 1];
      Mat *eye = matrix_eye(A->n, A->n);
      // s * I(:,:)
      matrix_scalar(eye, s);

      //A(:,:) - s * I(:,:)
      matrix_sub(A0, eye, mat_tmp);

      // qr(A(:,:) - s * I(:,:))
#ifdef NAIVE
      qr_householder(mat_tmp, R, Q);
#elif INTEL_MKL
      intel_mkl_qr(mat_tmp, R, Q);
#endif
      // A = R*Q + s*I
      matrix_add(matrix_mul(R, Q), eye, A0);
      Q1 = matrix_mul(Q1, Q);
      R = matrix_zeros(A0->m, A0->n);
      matrix_delete(eye);
      
      k++;
    }
    matrix_print(A0);
    matrix_delete(Q1);
    return matrix_diag(A0);
    */
  puts("A");
  matrix_print(A);
  complex *w = malloc(sizeof(complex) * A->n);
  Mat *z = matrix_new(A->m, A->n);
  LAPACKE_chseqr(LAPACK_ROW_MAJOR, 'E', 'I', A->n, 1, A->n, A->data, A->n, w, z->data, A->n);
  //qsort(wr, A->n, sizeof(*wr), my_compare);
  return w;

}


void eigen_values(Mat *A) {
  
    int nb_iter = 0;
    complex *v = malloc(A->n * sizeof(complex));
    for (int i = 0; i < A->n - 1; i++)
      v[i] = (complex){1, 0};
    float vNorm =  vect_norm(v, A->n);
    for (int i = 0; i < A->n; i++) {
        v[i].real /= vNorm;
    }
    vect_print(v, A->n);
    //iram(A, v, K, 20);
    
    Mat *Hm = matrix_zeros(N, N);
    Mat *Vm = matrix_zeros(A->n, N);
    complex *fm = malloc(sizeof(complex) * A->m);
    printf("Compute %d krylov space for Matrice A(%d, %d):\n", 20, A->m, A->n);
    float start = omp_get_wtime();
    
    arnoldi_iteration(A, v, 1, N, Hm, Vm, fm);
    float stop = omp_get_wtime();
    free(v);
    printf("Time : %lf\n", stop-start);
    int k = 0;
    puts("Hm:");
    matrix_print(Hm);
    puts("Vm: ");
    matrix_print(Vm);
/*    float *eigenValues = qr_alg_eigen(Hm);
    puts("After QR Algorithm:");
    matrix_print(Hm);
    puts("Eigen Value of Hm:");
    vect_print(eigenValues, Hm->m);
*/
}
/*
void double_qr_step(Mat *Hm) {
  int p = Hm->n;
  while (p > 2) {
    int q = p - 1;
    float s = Hm->data[q][q] + Hm->data[]
  }
}
*/

void iram(Mat *A, complex *v0, int k, int m) {
  Mat *Hm = matrix_zeros(m, m);
  Mat *Vm = matrix_zeros(A->n, m);
  complex *fm = malloc(sizeof(complex) * A->m);
  arnoldi_iteration(A, v0, 1, m, Hm, Vm, fm);
  int nb_iter = 0;
  while (nb_iter < 1) {
    printf("Iteration %d\n", nb_iter);
    complex *eigenValues = qr_alg_eigen(Hm);
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
    complex *fm_tmp = malloc(sizeof(complex) * A->m);
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
    matrix_copy_cond(tmp_2, Vm, k);
    complex *Vk = get_column(Vm, k);
    arnoldi_iteration(A, Vk, k, m, Hm, Vm, fm);
    nb_iter++;
  }
}
complex in[][3] = {
    {{2, 0},{1, 0}, {2, 0}},
    {{4, 0},{2, 0},{3, 0}},  
    {{3, 0},{2, 0},{2, 0}}
};
complex in2[][6] = {
    {{35,0},{1,0}, {6,0}, {26,0},{19,0},{24,0}},
    {{3,0}, {32,0}, {7,0},{21,0},{23,0},{25,0}},
    {{31,0},{9,0},{2,0},{22,0},{27,0},{20,0}},
    {{8,0},{28,0},{33,0},{17,0},{10,0},{15,0}},
    {{30,0},{5,0},{34,0},{12,0},{14,0},{16,0}},
    {{4,0},{36,0},{29,0},{13,0},{18,0},{11,0}}
};  
void init_random_matrix(Mat *A) {
  float tmp;
  srand(10);
  for (int i = 0; i < A->m * A->n; i++) {
    tmp = (float) (rand() % 100);
    A->data[i].real = tmp;
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
  for (int i = 0; i < N * N; i++)
    A->data[i] = in2[i / N][i % N];
  matrix_print(A);
  /*
  init_random_matrix(A);
  matrix_print(A);
  puts("");
  */
  float start = omp_get_wtime();
  eigen_values(A);
  float stop = omp_get_wtime();
 // printf("Time : %lf\n", stop-start);
  matrix_delete(A);
 
  return 0;
}



