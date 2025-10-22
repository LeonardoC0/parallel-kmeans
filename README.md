# Versão Atual - Implementação Sequencial do Kmeans

Para rodar:

g++ kmeans.cpp -o kmeans -fopenmp

 ./kmeans Sample.csv saida


Para rodar versão paralela mista (kmeans_parallel.cpp)

mpicxx -fopenmp -o kmeans_parallel kmeans_parallel.cpp

OMP_NUM_THREADS=4 (numero de threads) mpirun -np 1 (numero de processos) ./kmeans_parallel Sample.csv ./saida

LEMBRAR DE REMOVER OS PARENTESES, ESTÃO APENAS PARA INDICAR COMO MODIGICAR O TESTE:
1 processo com 4 threads:
OMP_NUM_THREADS=4 mpirun -np 1 ./kmeans_parallel Sample.csv ./saida
2 processos com 2 threads:
OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans_parallel Sample.csv ./saida
4 processos sem threads:
OMP_NUM_THREADS=1 mpirun -np 4 ./kmeans_parallel Sample.csv ./saida



