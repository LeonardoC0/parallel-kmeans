# Implementação de paralelismo do Algoritmo K-Means

O objetivo deste trabalho foi explorar a aplicação de Computação Paralela para otimizar o algoritmo de agrupamento K-Means. O estudo de caso utiliza o dataset "US Wildfire", com o intuito de agrupar focos de incêndio baseados em coordenadas geográficas.
A execução consiste em rodar o "Método do Cotovelo" (Elbow Method), executando o algoritmo iterativamente para valores de K variando de 2 a 20. 

## Alunos integrantes da equipe

* Daniel Valadares Souza
* Leonardo de Oliveira Carvalho
* Victor Monteiro Martinelli

  
# Instruções de utilização

## Implementação KMeans Sequencial:

Para rodar (kmeans.cpp)

g++ kmeans.cpp -o kmeans -fopenmp

 ./kmeans Sample.csv saida

# CPU
## Implementação KMeans Misto:

Para rodar versão paralela mista (kmeans_parallel.cpp)

mpicxx -fopenmp -o kmeans_parallel kmeans_parallel.cpp

OMP_NUM_THREADS={número de threads} mpirun -np {número de processos} ./kmeans_parallel Sample.csv ./saida

LEMBRAR DE REMOVER OS COLCHETES, ESTÃO APENAS PARA INDICAR COMO MODIFICAR O TESTE: <br>
1 processo com 4 threads: <br>
OMP_NUM_THREADS=4 mpirun -np 1 ./kmeans_parallel Sample.csv ./saida <br>
2 processos com 2 threads: <br>
OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans_parallel Sample.csv ./saida <br>
4 processos sem threads: <br>
OMP_NUM_THREADS=1 mpirun -np 4 ./kmeans_parallel Sample.csv ./saida <br>

## Implementação KMeans Misto 2:

mpicxx -O3 -fopenmp -march=native -o kmeans_mpi_bcast kmeans_mpi_bcast.cpp<br>
<br>
~/paralela$ # 1, 2, 4 e 8 threads<br>
OMP_NUM_THREADS=1 mpirun -np 1 ./kmeans_mpi_bcast Sample.csv ./saida<br>
OMP_NUM_THREADS=2 mpirun -np 1 ./kmeans_mpi_bcast Sample.csv ./saida<br>
OMP_NUM_THREADS=4 mpirun -np 1 ./kmeans_mpi_bcast Sample.csv ./saida<br>
OMP_NUM_THREADS=8 mpirun -np 1 ./kmeans_mpi_bcast Sample.csv ./saida<br>

# GPU

## Implementação KMeans OpenMP:

gcc8 -fopenmp -fopenmp-targets=nvptx64 -O3 kmeans_omp.cpp -o kmeans_omp
ou
nvcc -mp=gpu -O3 kmeans_omp.cpp -o kmeans_omp (compilador NVIDIA HPC)

./kmeans_omp Sample.csv ./saida


## Implementação KMeans CUDA:

nvcc -O3 kmeans_cuda.cu -o kmeans_cuda

./kmeans_cuda Sample.csv ./saida
