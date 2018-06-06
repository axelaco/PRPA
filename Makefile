test_bis: matrix.c
	gcc -fopenmp -o eigen_program_bis eigen_program_bis.c -lm -g3
test:
	gcc -fopenmp -o eigen_program eigen_program.c matrix.c -lm -g3
