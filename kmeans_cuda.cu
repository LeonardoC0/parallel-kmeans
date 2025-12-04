/**
 * nvcc -O3 kmeans_cuda.cu -o kmeans_cuda
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

// Para medir tempo sem OpenMP
#include <time.h>
double get_time() { return (double)clock() / CLOCKS_PER_SEC; }

using namespace std;

// --- KERNEL CUDA ---
// Executado na GPU: Cada thread calcula a distância de UM ponto para todos os clusters
__global__ void findNearestCluster(const double* points, const double* centroids, int* assignments, int* changed_flag, int n_points, int n_clusters, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_points) {
        double min_dist_sq = 1e30;
        int nearestClusterId = -1;

        for (int k = 0; k < n_clusters; k++) {
            double dist_sq = 0.0;
            for (int d = 0; d < dimensions; d++) {
                // Acesso linearizado: [ponto_id * dims + dimensão_atual]
                double p_val = points[idx * dimensions + d];
                double c_val = centroids[k * dimensions + d];
                double diff = p_val - c_val;
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearestClusterId = k + 1; // IDs 1-based
            }
        }

        if (assignments[idx] != nearestClusterId) {
            assignments[idx] = nearestClusterId;
            *changed_flag = 1; // Race condition benigna (apenas flag de "sujo")
        }
    }
}

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
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e') { tmp += line[i]; }
            else if (tmp.length() > 0) { try { values.push_back(stod(tmp)); } catch (...) {} tmp = ""; }
        }
        if (tmp.length() > 0) { try { values.push_back(stod(tmp)); } catch (...) {} }
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
    Cluster(int clusterId, const Point& centroid_p)
    {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid_p.getDimensions(); i++) this->centroid.push_back(centroid_p.getVal(i));
    }
    void addPoint(const Point& p) { points.push_back(p); }
    void removeAllPoints() { points.clear(); }
    int getId() const { return clusterId; }
    const Point& getPoint(int pos) const { return points[pos]; }
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
        for (int i = 0; i < K; i++) clusters[i].removeAllPoints();
    }

public:
    KMeans(int K, int iterations, string output_dir) : K(K), iters(iterations), output_dir(output_dir) {}
    
    double calculateWCSS() const
    {
        double total_wcss = 0.0;
        for (const auto& cluster : clusters) {
            double cluster_wcss = 0.0;
            for (int p = 0; p < cluster.getSize(); p++) {
                const Point& point = cluster.getPoint(p);
                double sum_sq_dist = 0.0;
                for (int d = 0; d < dimensions; d++) {
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
        if (total_points == 0) return;
        dimensions = all_points[0].getDimensions();
        clusters.clear();

        // Inicialização (CPU)
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

        // --- PREPARAÇÃO GPU (CUDA) ---
        
        // 1. Flattening dos pontos na CPU
        double* h_points_flat = new double[total_points * dimensions];
        int* h_assignments = new int[total_points];
        
        for(int i=0; i<total_points; i++) {
            h_assignments[i] = all_points[i].getCluster();
            for(int d=0; d<dimensions; d++) {
                h_points_flat[i * dimensions + d] = all_points[i].getVal(d);
            }
        }

        // 2. Alocação na GPU
        double *d_points, *d_centroids;
        int *d_assignments, *d_changed_flag;

        cudaMalloc(&d_points, total_points * dimensions * sizeof(double));
        cudaMalloc(&d_centroids, K * dimensions * sizeof(double));
        cudaMalloc(&d_assignments, total_points * sizeof(int));
        cudaMalloc(&d_changed_flag, sizeof(int));

        // 3. Cópia inicial (Pontos são estáticos)
        cudaMemcpy(d_points, h_points_flat, total_points * dimensions * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_assignments, h_assignments, total_points * sizeof(int), cudaMemcpyHostToDevice);

        double* h_centroids_flat = new double[K * dimensions];
        int iter = 1;
        bool changed_assignment = true;

        // Configuração Kernel
        int blockSize = 256;
        int numBlocks = (total_points + blockSize - 1) / blockSize;

        while (changed_assignment && iter <= iters)
        {
            changed_assignment = false;
            int h_changed_flag = 0;

            // A. Flattening dos centróides (CPU)
            for(int k=0; k<K; k++) {
                for(int d=0; d<dimensions; d++) {
                    h_centroids_flat[k * dimensions + d] = clusters[k].getCentroidByPos(d);
                }
            }

            // B. Copiar centróides para GPU
            cudaMemcpy(d_centroids, h_centroids_flat, K * dimensions * sizeof(double), cudaMemcpyHostToDevice);
            
            // Reset flag
            cudaMemcpy(d_changed_flag, &h_changed_flag, sizeof(int), cudaMemcpyHostToDevice);

            // C. Executar Kernel
            findNearestCluster<<<numBlocks, blockSize>>>(d_points, d_centroids, d_assignments, d_changed_flag, total_points, K, dimensions);
            cudaDeviceSynchronize();

            // D. Checar se houve mudança
            cudaMemcpy(&h_changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_changed_flag) {
                changed_assignment = true;
                
                // Trazer assignments de volta
                cudaMemcpy(h_assignments, d_assignments, total_points * sizeof(int), cudaMemcpyDeviceToHost);

                // E. Atualizar estrutura de classes na CPU
                for(int i=0; i<total_points; i++) {
                    all_points[i].setCluster(h_assignments[i]);
                }

                // F. Recalcular centróides (Lógica Original)
                clearClusters();
                for (int i = 0; i < total_points; i++) {
                    if (all_points[i].getCluster() > 0) {
                        clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
                    }
                }
                for (int i = 0; i < K; i++) {
                    int ClusterSize = clusters[i].getSize();
                    if (ClusterSize > 0) {
                        for (int j = 0; j < dimensions; j++) {
                            double sum = 0.0;
                            for (int p = 0; p < ClusterSize; p++) {
                                sum += clusters[i].getPoint(p).getVal(j);
                            }
                            clusters[i].setCentroidByPos(j, sum / ClusterSize);
                        }
                    }
                }
            }
            iter++;
        }

        // Limpeza GPU
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_assignments);
        cudaFree(d_changed_flag);

        // Limpeza Host
        delete[] h_points_flat;
        delete[] h_centroids_flat;
        delete[] h_assignments;

        if (K == 10) {
            ofstream pointsFile;
            pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
            for (int i = 0; i < total_points; i++) pointsFile << all_points[i].getCluster() << endl;
            pointsFile.close();
        }
    }
};

int main(int argc, char **argv)
{
    double start_time, end_time;
    start_time = get_time(); 

    if (argc != 3) {
        cout << "Error: Usage: ./kmeans_cuda <INPUT-FILE> <OUT-DIR>" << endl;
        return 1;
    }

    string output_dir = argv[2];
    string filename = argv[1];
    
    ifstream infile(filename.c_str());
    if (!infile.is_open()) { cout << "Error opening file." << endl; return 1; }

    int pointId = 1;
    vector<Point> all_points;
    string line;
    bool header = true;

    while (getline(infile, line))
    {
        if (header) { header = false; continue; }
        vector<string> cols;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) cols.push_back(cell);
        if (cols.size() < 5) continue; 

        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }
    infile.close();
    cout << "\nData fetched successfully! Total points: " << all_points.size() << endl;

    const int K_MIN = 2; 
    const int K_MAX = 20; 
    const int IT_MAX = 200; 

    if ((int)all_points.size() < K_MAX) cout << "Warning: Points < K_MAX." << endl;
    
    const vector<Point> original_points = all_points; 

    cout << "\n--- Running Elbow Method (GPU CUDA) ---\n" << endl;
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
    
    end_time = get_time();
    printf("\nTempo Total de Execução: %f segundos\n", end_time - start_time);

    return 0;
}