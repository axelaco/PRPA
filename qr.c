#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
typedef struct {
	int m, n;
	double ** v;
} mat_t, *mat;
 
mat matrix_new(int m, int n)
{
	mat x = malloc(sizeof(mat_t));
	x->v = malloc(sizeof(double*) * m);
	x->v[0] = calloc(sizeof(double), m * n);
	for (int i = 0; i < m; i++)
		x->v[i] = x->v[0] + n * i;
	x->m = m;
	x->n = n;
	return x;
}
 
void matrix_delete(mat m)
{
	free(m->v[0]);
	free(m->v);
	free(m);
}
 
void matrix_transpose(mat m)
{
	for (int i = 0; i < m->m; i++) {
		for (int j = 0; j < i; j++) {
			double t = m->v[i][j];
			m->v[i][j] = m->v[j][i];
			m->v[j][i] = t;
		}
	}
}
 
mat matrix_copy(int n, double a[][n], int m)
{
	mat x = matrix_new(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			x->v[i][j] = a[i][j];
	return x;
}

int is_diagonal(int n, mat x, int m){
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			if (i != j && x->v[i][j] != 0)
				return 0;
	return 1;
}
 
mat matrix_mul(mat x, mat y)
{
	if (x->n != y->m) return 0;
	mat r = matrix_new(x->m, y->n);
	for (int i = 0; i < x->m; i++)
		for (int j = 0; j < y->n; j++)
			for (int k = 0; k < x->n; k++)
				r->v[i][j] += x->v[i][k] * y->v[k][j];
	return r;
}
 
mat matrix_minor(mat x, int d)
{
	mat m = matrix_new(x->m, x->n);
	for (int i = 0; i < d; i++)
		m->v[i][i] = 1;
	for (int i = d; i < x->m; i++)
		for (int j = d; j < x->n; j++)
			m->v[i][j] = x->v[i][j];
	return m;
}
 
/* c = a + b * s */
double *vmadd(double a[], double b[], double s, double c[], int n)
{
	for (int i = 0; i < n; i++)
		c[i] = a[i] + s * b[i];
	return c;
}
 
/* m = I - v v^T */
mat vmul(double v[], int n)
{
	mat x = matrix_new(n, n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			x->v[i][j] = -2 *  v[i] * v[j];
	for (int i = 0; i < n; i++)
		x->v[i][i] += 1;
 
	return x;
}
 
/* ||x|| */
double vnorm(double x[], int n)
{
	double sum = 0;
	for (int i = 0; i < n; i++) sum += x[i] * x[i];
	return sqrt(sum);
}
 
/* y = x / d */
double* vdiv(double x[], double d, double y[], int n)
{
	for (int i = 0; i < n; i++) y[i] = x[i] / d;
	return y;
}
 
/* take c-th column of m, put in v */
double* mcol(mat m, double *v, int c)
{
	for (int i = 0; i < m->m; i++)
		v[i] = m->v[i][c];
	return v;
}
 
void matrix_show(mat m)
{
	for(int i = 0; i < m->m; i++) {
		for (int j = 0; j < m->n; j++) {
			printf(" %8.3f", m->v[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
 
void householder(mat m, mat *R, mat *Q)
{
	mat q[m->m];
	mat z = m, z1;
	for (int k = 0; k < m->n && k < m->m - 1; k++) {
		double e[m->m], x[m->m], a;
		z1 = matrix_minor(z, k);
		if (z != m) matrix_delete(z);
		z = z1;
 
		mcol(z, x, k);
		a = vnorm(x, m->m);
		if (m->v[k][k] > 0) a = -a;
 
		for (int i = 0; i < m->m; i++)
			e[i] = (i == k) ? 1 : 0;
 
		vmadd(x, e, a, e, m->m);
		vdiv(e, vnorm(e, m->m), e, m->m);
		q[k] = vmul(e, m->m);
		z1 = matrix_mul(q[k], z);
		if (z != m) matrix_delete(z);
		z = z1;
	}
	matrix_delete(z);
	*Q = q[0];
	*R = matrix_mul(q[0], m);
	for (int i = 1; i < m->n && i < m->m - 1; i++) {
		z1 = matrix_mul(q[i], *Q);
		if (i > 1) matrix_delete(*Q);
		*Q = z1;
		matrix_delete(q[i]);
	}
	matrix_delete(q[0]);
	z = matrix_mul(*Q, m);
	matrix_delete(*R);
	*R = z;
	matrix_transpose(*Q);
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

int is_triangle_sup(mat X){
	for (int i = 1; i < X->m; i++)
		for (int j = 0; j < i; j++)
			if (fabs(X->v[i][j]) > 0.0005)
				return 0;
	return 1;
}

int main()
{
	mat R, Q, X;
	mat x = matrix_copy(10, in, 10);
	householder(x, &R, &Q);

	int nb_iter = 0;
	puts("Q"); matrix_show(Q);
	puts("R"); matrix_show(R);

       	X = matrix_mul(Q, R);
	while (is_triangle_sup(X) == 0){
		householder(X, &R, &Q);
		X = matrix_mul(R, Q);
		nb_iter++;
	}	
	// to show their product is the input matrix
	puts("Q end"); matrix_show(Q);
	puts("R end"); matrix_show(R);
	printf("X (%d iter)\n", nb_iter); matrix_show(X);
	printf("Eigen values are: ");
	for (int i = 0; i < X->m; i++)
		printf(" %f ", X->v[i][i]);
	printf("\n");

	matrix_delete(x);
	matrix_delete(R);
	matrix_delete(Q);
	matrix_delete(X);
	//matrix_delete(m);
	return 0;
}
