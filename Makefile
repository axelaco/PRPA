test_bis: matrix.c
	gcc -fopenmp -o eigen_program_bis eigen_program_bis.c -lm -g3
programf:
	gcc -fopenmp -o eigen_program_f eigen_programf.c math.c -lm -g3
program:
	gcc -fopenmp -o eigen_program eigen_program.c matrix.c -lm -g3 -llapack
