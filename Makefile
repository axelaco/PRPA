test: matrix.c
	gcc -fopenmp -o eigen_program eigen_program.c matrix.c -lm
