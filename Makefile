program:
	gcc -fopenmp -o eigen_program_mkl eigen_program.c math.c -lm -g3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
