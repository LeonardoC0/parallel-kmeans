#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>  
#include <iomanip> 
#include <mpi.h>



#ifdef _OPENMP

#include <omp.h>

#else

double omp_get_wtime() { return (double)clock() / CLOCKS_PER_SEC; }

#endif



using namespace std;



/* TEMPO DE EXECUÇÕES

    1 processo com 4 threads:

    Tempo Total de Execução: 38.664712 segundos (para todas as iterações de K)

    2 processos com 2 threads:

    Tempo Total de Execução: 19.928303 segundos (para todas as iterações de K)

    4 processos sem threads:

    Tempo Total de Execução: 11.586576 segundos (para todas as iterações de K)

*/



// Classe Point (inalterada)

class Point

{

private:

    int pointId, clusterId;

    int dimensions;

    vector<double> values;



    vector<double> lineToVec(string &line)

    {

        vector<double> values;

        string tmp = "";



        // Usando stringstream para parsear a linha de forma mais robusta

        stringstream ss(line);

        string segment;



        for (int i = 0; i < (int)line.length(); i++)

        {

            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')

            {

                tmp += line[i];

            }

            else if (tmp.length() > 0)

            {

                try

                {

                    values.push_back(stod(tmp));

                }

                catch (...)

                {

                    // Ignora conversão inválida (como strings vazias)

                }

                tmp = "";

            }

        }

        if (tmp.length() > 0)

        {

            try

            {

                values.push_back(stod(tmp));

            }

            catch (...)

            {

                // Ignora conversão inválida

            }

        }

        return values;

    }



public:

    Point(int id, string line)

    {

        pointId = id;

        values = lineToVec(line);

        dimensions = values.size();

        clusterId = 0; // Inicialmente não atribuído a nenhum cluster

    }



    int getDimensions() const { return dimensions; }

    int getCluster() const { return clusterId; }

    int getID() const { return pointId; }



    void setCluster(int val) { clusterId = val; }



    double getVal(int pos) const { return values[pos]; }

};



// Classe Cluster (inalterada, exceto pela adição de 'const' em getters)

class Cluster

{

private:

    int clusterId;

    vector<double> centroid;

    vector<Point> points;



public:

    Cluster(int clusterId, const Point &centroid_p)

    {

        this->clusterId = clusterId;

        for (int i = 0; i < centroid_p.getDimensions(); i++)

        {

            this->centroid.push_back(centroid_p.getVal(i));

        }

        // Não adicionamos o ponto aqui, pois o ponto inicial é apenas o centróide.

        // A adição real ocorre no loop de iteração do K-Means.

    }



    void addPoint(const Point &p)

    {

        // A atribuição do cluster ID para 'p' deve ser feita no KMeans::run

        // antes de chamar addPoint.

        points.push_back(p);

    }



    void removeAllPoints() { points.clear(); }



    int getId() const { return clusterId; }



    const Point &getPoint(int pos) const { return points[pos]; }



    int getSize() const { return points.size(); }



    double getCentroidByPos(int pos) const { return centroid[pos]; }



    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }

};



class KMeans

{

private:

    int K, iters, dimensions, total_points;

    vector<Cluster> clusters;

    string output_dir;



    void clearClusters()

    {

        for (int i = 0; i < K; i++)

        {

            clusters[i].removeAllPoints();

        }

    }



    // Calcula a distância euclidiana ao quadrado

    double calculateSquaredDistance(const Point &point, const Cluster &cluster)

    {

        double sum = 0.0;

        // Tentativa de paralelizar usando "#pragma omp parallel for reduction(+:sum)" não sucessedida, overhead muito alto

        for (int i = 0; i < dimensions; i++)

        {

            double diff = cluster.getCentroidByPos(i) - point.getVal(i);

            sum += diff * diff;

        }

        return sum;

    }



    int getNearestClusterId(const Point &point)

    {

        // Usamos a distância euclidiana ao quadrado para evitar a raiz quadrada (otimização)

        double min_dist_sq = calculateSquaredDistance(point, clusters[0]);

        int NearestClusterId = clusters[0].getId();

        // Não é possivel paralelizar por conta do alto overhead

        for (int i = 1; i < K; i++)

        {

            double dist_sq = calculateSquaredDistance(point, clusters[i]);



            if (dist_sq < min_dist_sq)

            {

                min_dist_sq = dist_sq;

                NearestClusterId = clusters[i].getId();

            }

        }

        return NearestClusterId;

    }



public:

    KMeans(int K, int iterations, string output_dir)

    {

        this->K = K;

        this->iters = iterations;

        this->output_dir = output_dir;

    }



    // Novo método para calcular o WCSS (Within-Cluster Sum of Squares)

    double calculateWCSS() const

    {

        double total_wcss = 0.0;

        // paralelizar usando o "#pragma omp parallel for reduction(+:total_wcss) concentrando as somas em total_wcss"

        #pragma omp parallel for reduction(+ : total_wcss)

        for (const auto &cluster : clusters)

        {

            double cluster_wcss = 0.0;

            for (int p = 0; p < cluster.getSize(); p++)

            {

                // Calcula a distância euclidiana ao quadrado do ponto ao seu centróide

                const Point &point = cluster.getPoint(p);

                double sum_sq_dist = 0.0;



                for (int d = 0; d < dimensions; d++)

                {

                    double diff = point.getVal(d) - cluster.getCentroidByPos(d);

                    sum_sq_dist += diff * diff;

                }

                cluster_wcss += sum_sq_dist;

            }

            total_wcss += cluster_wcss;

        }

        return total_wcss;

    }



    // O método run agora não grava arquivos de saída, apenas executa o algoritmo.

    void run(vector<Point> &all_points)

    {

        total_points = all_points.size();

        if (total_points == 0)

            return;

        dimensions = all_points[0].getDimensions();

        clusters.clear(); // Garante que clusters está vazio antes de iniciar a execução



        // Inicializando Clusters (usando pontos aleatórios)

        vector<int> used_pointIndices;

        srand(time(0) + K); // Semente para inicialização diferente para cada K



        for (int i = 1; i <= K; i++)

        {

            while (true)

            {

                int index = rand() % total_points;



                // Evita selecionar o mesmo ponto como centróide inicial

                if (find(used_pointIndices.begin(), used_pointIndices.end(), index) == used_pointIndices.end())

                {

                    used_pointIndices.push_back(index);

                    // O ponto inicial é apenas um centróide temporário para a classe Cluster

                    Cluster cluster(i, all_points[index]);

                    clusters.push_back(cluster);

                    break;

                }

            }

        }

        // cout << "Clusters initialized = " << clusters.size() << endl << endl;



        int iter = 1;

        bool changed_assignment = true;

        while (changed_assignment && iter <= iters)

        {

            changed_assignment = false;

            // cout << "Iter - " << iter << "/" << iters << endl;



            // 1. Atribuição de pontos ao cluster mais próximo

            for (int i = 0; i < total_points; i++)

            {

                int currentClusterId = all_points[i].getCluster();

                int nearestClusterId = getNearestClusterId(all_points[i]);



                if (currentClusterId != nearestClusterId)

                {

                    all_points[i].setCluster(nearestClusterId);

                    changed_assignment = true;

                }

            }



            // 2. Limpar todos os clusters existentes e reatribuir pontos

            clearClusters();

            for (int i = 0; i < total_points; i++)

            {

                // cluster index é ID-1

                // Verificação de segurança: se o clusterId for 0 (ponto não atribuído),

                // ele não será adicionado ao cluster.

                if (all_points[i].getCluster() > 0)

                {

                    clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);

                }

            }



            // 3. Recalcular o centróide de cada cluster

            //Paralelizamos, não precisamos informar uma variavel para redução pois são alteradas posições diferentes de cluster[i]

            #pragma omp parallel for

            for (int i = 0; i < K; i++)

            {

                int ClusterSize = clusters[i].getSize();



                // Recalcula o centróide apenas se houver pontos no cluster

                if (ClusterSize > 0)

                {

                    for (int j = 0; j < dimensions; j++)

                    {

                        double sum = 0.0;

                        for (int p = 0; p < ClusterSize; p++)

                        {

                            sum += clusters[i].getPoint(p).getVal(j);

                        }

                        clusters[i].setCentroidByPos(j, sum / ClusterSize);

                    }

                }

            }



            if (!changed_assignment)

            {

                // cout << "Clustering completed by convergence in iteration : " << iter << endl;

                break;

            }

            iter++;

        }

        // cout << "Clustering completed for K=" << K << endl << endl;



        // Opcional: Gravar o resultado do último K executado (K=10 neste caso)

        if (K == 10)

        {

            ofstream pointsFile;

            pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);

            for (int i = 0; i < total_points; i++)

            {

                pointsFile << all_points[i].getCluster() << endl;

            }

            pointsFile.close();

        }

    }

};



int main(int argc, char **argv)

{



    // MODIFICAÇÃO MPI: Inicialização do ambiente MPI

    int rank, world_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);



    double start_time, end_time;

    // MODIFICAÇÃO MPI: Apenas o rank 0 mede o tempo

    if (rank == 0)

    {

        start_time = omp_get_wtime();

    }



    if (argc != 3)



    {

        // MODIFICAÇÃO MPI: Apenas o rank 0 imprime o erro

        if (rank == 0)

        {

            cout << "Error: command-line argument count mismatch. \n Usage: ./elbow_kmeans <INPUT-FILE> <OUT-DIR>" << endl;

        }

        // MODIFICAÇÃO MPI: Todos os processos devem finalizar antes de sair

        MPI_Finalize();

        return 1;

    }



    string output_dir = argv[2];

    string filename = argv[1];



    // --- Leitura e Preparação dos Dados ---

    // MODIFICAÇÃO MPI: Todos os processos leem o arquivo.

    // Isso é redundante, mas é muito mais simples do que

    // transmitir a estrutura de dados 'vector<Point>' (que contém vetores)

    // do rank 0 para todos os outros.

    ifstream infile(filename.c_str());



    if (!infile.is_open())



    {

        if (rank == 0)

        {

            cout << "Error: Failed to open file '" << filename << "'." << endl;

        }

        MPI_Finalize();

        return 1;

    }



    // ... [código de leitura de arquivo inalterado] ...

    int pointId = 1;

    vector<Point> all_points;

    string line;

    bool header = true;

    while (getline(infile, line))



    {

        if (header)

        {

            header = false;

            continue;

        }

        vector<string> cols;

        stringstream ss(line);

        string cell;

        while (getline(ss, cell, ','))

            cols.push_back(cell);

        if (cols.size() < 5)

            continue;

        Point point(pointId, line);

        all_points.push_back(point);

        pointId++;

    }

    infile.close();

    // MODIFICAÇÃO MPI: Apenas rank 0 imprime

    if (rank == 0)

    {

        cout << "\nData fetched successfully! Total points: " << all_points.size() << endl;

    }



    if (all_points.empty())

    {

        if (rank == 0)

        {

            cout << "Error: Input file is empty or data could not be parsed." << endl;

        }

        MPI_Finalize();

        return 1;

    }



    // --- Implementação do Método do Cotovelo ---

    const int K_MIN = 2;

    const int K_MAX = 20;

    const int IT_MAX = 200;



    if ((int)all_points.size() < K_MAX)



    {

        if (rank == 0)

        { // Apenas rank 0 imprime o aviso

            cout << "Warning: Number of points is less than K_MAX. Reducing K_MAX to " << all_points.size() - 1 << endl;

        }

    }



    const vector<Point> original_points = all_points;



    // MODIFICAÇÃO MPI: Cada processo armazena seus próprios resultados

    vector<pair<int, double>> local_wcss_results;



    if (rank == 0)

    {

        cout << "\n--- Running Elbow Method (K=" << K_MIN << " to K=" << K_MAX << ") ---\n"

             << endl;

        cout << "K | WCSS (Within-Cluster Sum of Squares)\n";

        cout << "------------------------------------------\n";

    }



    // MODIFICAÇÃO MPI: Paralelização do loop 'K' com MPI (Distribuição Cíclica)

    // Cada processo (rank) começa em 'K_MIN + rank' e pula de 'world_size' em 'world_size'.

    // Ex: 4 processos:

    // Rank 0: K = 2, 6, 10, 14, 18

    // Rank 1: K = 3, 7, 11, 15, 19

    // Rank 2: K = 4, 8, 12, 16, 20

    // Rank 3: K = 5, 9, 13, 17

    for (int K = K_MIN + rank; K <= K_MAX && K < (int)original_points.size(); K += world_size)



    {

        vector<Point> current_points = original_points;



        // MODIFICAÇÃO OMP: Esta chamada 'kmeans.run()' usará threads OpenMP internamente.

        KMeans kmeans(K, IT_MAX, output_dir);

        kmeans.run(current_points);



        double wcss = kmeans.calculateWCSS();



        // MODIFICAÇÃO MPI: Salva o resultado no vetor local

        local_wcss_results.push_back({K, wcss});



        // MODIFICAÇÃO MPI: Removemos a impressão de dentro do loop

        // para evitar saída desordenada. Apenas o rank 0 imprimirá no final.

    }



    // --- MODIFICAÇÃO MPI: Coleta e Análise dos Resultados ---



    if (rank == 0)

    {

        // O Rank 0 começa com seus próprios resultados

        vector<pair<int, double>> global_wcss_results = local_wcss_results;



        // Recebe os resultados dos outros processos

        for (int p = 1; p < world_size; p++)

        {

            int recv_count;

            // 1. Recebe quantos resultados o processo 'p' tem

            MPI_Recv(&recv_count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



            if (recv_count > 0)

            {

                // Aloca espaço para os dados serializados

                vector<int> k_vals(recv_count);

                vector<double> wcss_vals(recv_count);



                // 2. Recebe o vetor de 'K'

                MPI_Recv(k_vals.data(), recv_count, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 3. Recebe o vetor de 'WCSS'

                MPI_Recv(wcss_vals.data(), recv_count, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



                // 4. "Zipa" os dados de volta para o vetor global

                for (int i = 0; i < recv_count; i++)

                {

                    global_wcss_results.push_back({k_vals[i], wcss_vals[i]});

                }

            }

        }



        // Ordena os resultados globais por K

        sort(global_wcss_results.begin(), global_wcss_results.end());



        // Imprime a tabela final

        for (const auto &pair : global_wcss_results)

        {

            cout << fixed << setprecision(4) << pair.first << " | " << pair.second << endl;

        }



        cout << "\n=======================================================\n";

        cout << "RESULTADOS DO MÉTODO DO COTOVELO:\n";

        cout << "O 'K' ideal (o cotovelo) é o ponto onde o valor do WCSS para de diminuir rapidamente.\n";



        end_time = omp_get_wtime();

        printf("\nTempo Total de Execução: %f segundos (para todas as iterações de K)\n", end_time - start_time);

    }

    else

    {

        // Processos "trabalhadores" (rank > 0) enviam seus resultados para o rank 0

        int send_count = local_wcss_results.size();



        // 1. Envia a contagem de resultados

        MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);



        if (send_count > 0)

        {

            // "Unzipa" os dados para enviar

            vector<int> k_vals(send_count);

            vector<double> wcss_vals(send_count);

            for (int i = 0; i < send_count; i++)

            {

                k_vals[i] = local_wcss_results[i].first;

                wcss_vals[i] = local_wcss_results[i].second;

            }



            // 2. Envia o vetor de 'K'

            MPI_Send(k_vals.data(), send_count, MPI_INT, 0, 1, MPI_COMM_WORLD);

            // 3. Envia o vetor de 'WCSS'

            MPI_Send(wcss_vals.data(), send_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        }

    }



    // MODIFICAÇÃO MPI: Finaliza o ambiente MPI

    MPI_Finalize();

    return 0;

}
