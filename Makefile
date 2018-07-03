program_mkl:
	gcc -fopenmp -o eigen_program_mkl -DINTEL_MKL eigen_program.c math.c -lm -g3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
program_naive:
	gcc -fopenmp -o eigen_program -DNAIVE eigen_program.c math.c -lm -g3
program_cpx_mkl:
	gcc -fopenmp -o eigen_program_cpx_mkl -DINTEL_MKL eigen_program_complex.c math_complex.c -lm -g3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
program_cpx_naive:
	gcc -fopenmp -o eigen_program_cpx -DNAIVE eigen_program_complex.c math_complex.c -lm -g3
program_test:
	gcc -fopenmp -o test_complex -DNAIVE test_complex_func.c math_complex.c -lm -g3
program_test_mkl:
	gcc -fopenmp -o test_complex_mkl -DINTEL_MKL test_complex_func.c math_complex.c -lm -g3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
