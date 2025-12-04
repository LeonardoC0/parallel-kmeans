/**
 * gcc8 -fopenmp -fopenmp-targets=nvptx64 -O3 kmeans_omp.cpp -o kmeans_omp
 * ou
 * nvc++ -mp=gpu -O3 kmeans_omp.cpp -o kmeans_omp (compilador NVIDIA HPC)
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <omp.h>

using namespace std;

// --- CLASSE POINT (Mantida Original) ---
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
        clusterId = 0;
    }

    int getDimensions() const { return dimensions; }
    int getCluster() const { return clusterId; }
    int getID() const { return pointId; }
    void setCluster(int val) { clusterId = val; }
    double getVal(int pos) const { return values[pos]; }
};

// --- CLASSE CLUSTER (Mantida Original) ---
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
    }

    void addPoint(const Point &p) { points.push_back(p); }
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

public:
    KMeans(int K, int iterations, string output_dir)
    {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
    }

    double calculateWCSS() const
    {
        double total_wcss = 0.0;
        for (const auto &cluster : clusters)
        {
            double cluster_wcss = 0.0;
            for (int p = 0; p < cluster.getSize(); p++)
            {
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

    void run(vector<Point> &all_points)
    {
        total_points = all_points.size();
        if (total_points == 0)
            return;
        dimensions = all_points[0].getDimensions();
        clusters.clear();

        // Inicializando Clusters (CPU)
        vector<int> used_pointIndices;
        srand(time(0) + K);

        for (int i = 1; i <= K; i++)
        {
            while (true)
            {
                int index = rand() % total_points;
                if (find(used_pointIndices.begin(), used_pointIndices.end(), index) == used_pointIndices.end())
                {
                    used_pointIndices.push_back(index);
                    Cluster cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }

        // --- PREPARAÇÃO DE DADOS PARA GPU (FLATTENING) ---
        // Achatamos os objetos em arrays contíguos para a GPU processar
        double *points_flat = new double[total_points * dimensions];
        int *assignments = new int[total_points];
        double *centroids_flat = new double[K * dimensions];

        // Copia dados dos objetos Point para o array plano
        for (int i = 0; i < total_points; i++)
        {
            assignments[i] = all_points[i].getCluster();
            for (int d = 0; d < dimensions; d++)
            {
                points_flat[i * dimensions + d] = all_points[i].getVal(d);
            }
        }

        int iter = 1;
        bool changed_assignment = true;

// Mapeamento de dados para a GPU
// points_flat: map(to) -> enviados apenas uma vez
// assignments: map(alloc) -> alocados lá, depois copiados de volta
// centroids_flat: map(alloc) -> alocados lá, atualizados a cada iteração
#pragma omp target data map(to : points_flat[0 : total_points * dimensions]) \
    map(alloc : assignments[0 : total_points], centroids_flat[0 : K * dimensions])
        {
            while (changed_assignment && iter <= iters)
            {
                changed_assignment = false;

                // 1. Atualizar array plano de centróides (Baseado nos objetos Cluster da CPU)
                for (int k = 0; k < K; k++)
                {
                    for (int d = 0; d < dimensions; d++)
                    {
                        centroids_flat[k * dimensions + d] = clusters[k].getCentroidByPos(d);
                    }
                }

// Envia centróides novos para GPU
#pragma omp target update to(centroids_flat[0 : K * dimensions])

                int changed_flag = 0; // Flag simples para detecção de mudança

// --- KERNEL PRINCIPAL NA GPU ---
#pragma omp target teams distribute parallel for map(tofrom : changed_flag)
                for (int i = 0; i < total_points; i++)
                {
                    double min_dist_sq = 1e30; // Valor alto
                    int nearestClusterId = -1;

                    // Busca o cluster mais próximo para o ponto i
                    for (int k = 0; k < K; k++)
                    {
                        double dist_sq = 0.0;
                        for (int d = 0; d < dimensions; d++)
                        {
                            double diff = points_flat[i * dimensions + d] - centroids_flat[k * dimensions + d];
                            dist_sq += diff * diff;
                        }

                        if (dist_sq < min_dist_sq)
                        {
                            min_dist_sq = dist_sq;
                            nearestClusterId = k + 1; // ID 1-based
                        }
                    }

                    if (assignments[i] != nearestClusterId)
                    {
                        assignments[i] = nearestClusterId;
                        changed_flag = 1; // Marca que houve mudança
                    }
                }

                if (changed_flag)
                    changed_assignment = true;

                // Se houve mudança, precisamos trazer os assignments de volta para re-calcular as médias na CPU
                if (changed_assignment)
                {
#pragma omp target update from(assignments[0 : total_points])

                    // 2. Atualizar objetos Point na CPU com os resultados da GPU
                    for (int i = 0; i < total_points; i++)
                    {
                        all_points[i].setCluster(assignments[i]);
                    }

                    // 3. Recalcular centróides (Lógica Original da CPU)
                    clearClusters();
                    for (int i = 0; i < total_points; i++)
                    {
                        if (all_points[i].getCluster() > 0)
                        {
                            clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
                        }
                    }

                    for (int i = 0; i < K; i++)
                    {
                        int ClusterSize = clusters[i].getSize();
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
                }
                iter++;
            }
        } // Fim do target data

        // Limpeza de memória auxiliar
        delete[] points_flat;
        delete[] assignments;
        delete[] centroids_flat;

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
    double start_time, end_time;
    start_time = omp_get_wtime();

    if (argc != 3)
    {
        cout << "Error: command-line argument count mismatch. \n Usage: ./kmeans_omp <INPUT-FILE> <OUT-DIR>" << endl;
        return 1;
    }

    string output_dir = argv[2];
    string filename = argv[1];

    ifstream infile(filename.c_str());
    if (!infile.is_open())
    {
        cout << "Error: Failed to open file '" << filename << "'." << endl;
        return 1;
    }

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
    cout << "\nData fetched successfully! Total points: " << all_points.size() << endl;

    const int K_MIN = 2;
    const int K_MAX = 20;
    const int IT_MAX = 200;

    if ((int)all_points.size() < K_MAX)
    {
        cout << "Warning: Number of points is less than K_MAX." << endl;
    }

    const vector<Point> original_points = all_points;

    cout << "\n--- Running Elbow Method (GPU OpenMP) ---\n"
         << endl;
    cout << "K | WCSS \n";
    cout << "---------\n";

    for (int K = K_MIN; K <= K_MAX && K < (int)original_points.size(); ++K)
    {
        vector<Point> current_points = original_points;
        KMeans kmeans(K, IT_MAX, output_dir);
        kmeans.run(current_points);
        double wcss = kmeans.calculateWCSS();
        cout << fixed << setprecision(4) << K << " | " << wcss << endl;
    }

    end_time = omp_get_wtime();
    printf("\nTempo Total de Execução: %f segundos\n", end_time - start_time);

    return 0;

}
