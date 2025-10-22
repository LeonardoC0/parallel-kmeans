# Versão Atual - Implementação Sequencial do Kmeans

Para rodar:

g++ kmeans.cpp -o kmeans -fopenmp

 ./kmeans Sample.csv saida


Para rodar versão paralela mista (kmeans_parallel.cpp)

mpicxx -fopenmp -o kmeans_parallel kmeans_parallel.cpp

OMP_NUM_THREADS=4 mpirun -np 1 ./kmeans_parallel Sample.csv ./saida
numero de threads             numero de processos
