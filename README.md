#Implementação KMeans Sequencial:

Para rodar:

g++ kmeans.cpp -o kmeans -fopenmp

 ./kmeans Sample.csv saida

#Implementação KMeans Misto:

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

#Implementação KMeans Misto 2:





