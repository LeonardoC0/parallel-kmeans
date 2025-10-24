/*
Hybrid MPI/OpenMP K-Means with Elbow Method (broadcast-based data distribution)
-------------------------------------------------------------------------------
This version avoids having every MPI rank read and parse the CSV file.
Rank 0 reads the CSV, parses into a flat double array, then broadcasts:
  - number of points (N)
  - number of dimensions (D)
  - the raw data values (N*D doubles)
All ranks reconstruct Point objects from the broadcast data.

We keep the "cyclic K distribution" across ranks (2..K_MAX), and use OpenMP
inside each rank for centroid recomputation and WCSS. The assignment step
remains serial per point (vectorized or further OMP is not ideal due to overhead
and memory access patterns, but can be explored).

Timing notes:
- Only rank 0 measures and prints total runtime for the full Elbow loop.
- Insert your measured times (parcode server) in the comment banner below
  to satisfy item (v) of the assignment.
-------------------------------------------------------------------------------

TIMES (parcode) — fill with your real measurements:
  1 process with 4 threads:   Tempo Total de Execução: ________ s
  2 processes with 2 threads: Tempo Total de Execução: ________ s
  4 processes with 1 thread:  Tempo Total de Execução: ________ s

Compile:
  mpicxx -O3 -fopenmp -march=native -o kmeans_mpi_bcast kmeans_mpi_bcast.cpp

Run examples:
  # 1 process, 4 threads
  OMP_NUM_THREADS=4 mpirun -np 1 ./kmeans_mpi_bcast Sample.csv ./saida

  # 2 processes, 2 threads each
  OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans_mpi_bcast Sample.csv ./saida

  # 4 processes, no threads
  OMP_NUM_THREADS=1 mpirun -np 4 ./kmeans_mpi_bcast Sample.csv ./saida
*/

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

// ------------------------------- Point ---------------------------------
class Point {
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    // Parses free-form numeric line (numbers separated by any non-numeric char)
    vector<double> lineToVec(string &line)
    {
        vector<double> values;
        string tmp = "";
        for (int i = 0; i < (int)line.length(); i++) {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e') {
                tmp += line[i];
            } else if (tmp.length() > 0) {
                try { values.push_back(stod(tmp)); } catch (...) {}
                tmp = "";
            }
        }
        if (tmp.length() > 0) {
            try { values.push_back(stod(tmp)); } catch (...) {}
        }
        return values;
    }

public:
    Point() : pointId(0), clusterId(0), dimensions(0) {}
    Point(int id, string line)
    {
        pointId = id;
        values = lineToVec(line);
        dimensions = (int)values.size();
        clusterId = 0;
    }
    // New constructor from raw array slice (used after MPI_Bcast)
    Point(int id, const double* raw, int D)
    {
        pointId = id;
        dimensions = D;
        values.assign(raw, raw + D);
        clusterId = 0;
    }

    int getDimensions() const { return dimensions; }
    int getCluster()    const { return clusterId; }
    int getID()         const { return pointId; }

    void setCluster(int val)  { clusterId = val; }
    double getVal(int pos) const { return values[pos]; }
};

// ------------------------------ Cluster --------------------------------
class Cluster {
private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, const Point &centroid_p)
    {
        this->clusterId = clusterId;
        centroid.reserve(centroid_p.getDimensions());
        for (int i = 0; i < centroid_p.getDimensions(); i++) {
            centroid.push_back(centroid_p.getVal(i));
        }
    }

    void addPoint(const Point &p) { points.push_back(p); }
    void removeAllPoints() { points.clear(); }

    int getId() const { return clusterId; }
    const Point &getPoint(int pos) const { return points[pos]; }
    int getSize() const { return (int)points.size(); }

    double getCentroidByPos(int pos) const { return centroid[pos]; }
    void setCentroidByPos(int pos, double val) { centroid[pos] = val; }
};

// ------------------------------- KMeans --------------------------------
class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;

    void clearClusters() {
        for (int i = 0; i < K; i++) clusters[i].removeAllPoints();
    }

    // Euclidean squared distance
    inline double calculateSquaredDistance(const Point &point, const Cluster &cluster) const {
        double sum = 0.0;
        for (int i = 0; i < dimensions; i++) {
            double diff = cluster.getCentroidByPos(i) - point.getVal(i);
            sum += diff * diff;
        }
        return sum;
    }

    int getNearestClusterId(const Point &point) const {
        double min_dist_sq = calculateSquaredDistance(point, clusters[0]);
        int nearestId = clusters[0].getId();
        for (int i = 1; i < K; i++) {
            double dist_sq = calculateSquaredDistance(point, clusters[i]);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearestId = clusters[i].getId();
            }
        }
        return nearestId;
    }

public:
    KMeans(int K, int iterations, string output_dir)
    : K(K), iters(iterations), output_dir(output_dir) {}

    double calculateWCSS() const
    {
        double total_wcss = 0.0;
        #pragma omp parallel for reduction(+:total_wcss) schedule(static)
        for (int ci = 0; ci < (int)clusters.size(); ++ci) {
            const Cluster &cluster = clusters[ci];
            double cluster_wcss = 0.0;
            for (int p = 0; p < cluster.getSize(); p++) {
                const Point &point = cluster.getPoint(p);
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
        total_points = (int)all_points.size();
        if (total_points == 0) return;
        dimensions = all_points[0].getDimensions();
        clusters.clear();

        // Init centroids from random unique points (seed per K for reproducibility)
        vector<int> used;
        srand(1234 + K);
        for (int i = 1; i <= K; i++) {
            while (true) {
                int idx = rand() % total_points;
                if (find(used.begin(), used.end(), idx) == used.end()) {
                    used.push_back(idx);
                    clusters.emplace_back(i, all_points[idx]);
                    break;
                }
            }
        }

        int iter = 1;
        bool changed = true;
        while (changed && iter <= iters) {
            changed = false;

            // 1) Assign
            for (int i = 0; i < total_points; i++) {
                int curr = all_points[i].getCluster();
                int near = getNearestClusterId(all_points[i]);
                if (curr != near) {
                    all_points[i].setCluster(near);
                    changed = true;
                }
            }

            // 2) Rebuild cluster membership
            clearClusters();
            for (int i = 0; i < total_points; i++) {
                if (all_points[i].getCluster() > 0)
                    clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            // 3) Recompute centroids (OpenMP)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < K; i++) {
                int csize = clusters[i].getSize();
                if (csize > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        double sum = 0.0;
                        for (int p = 0; p < csize; p++) sum += clusters[i].getPoint(p).getVal(j);
                        clusters[i].setCentroidByPos(j, sum / csize);
                    }
                }
            }

            if (!changed) break;
            iter++;
        }

        // Optional dump for K==10
        if (K == 10) {
            ofstream f(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
            for (int i = 0; i < total_points; i++) f << all_points[i].getCluster() << "\n";
            f.close();
        }
    }
};

// ---------------------------- CSV Utilities ----------------------------
static bool read_csv_rank0(const string &filename, vector<double> &flat, int &N, int &D)
{
    ifstream infile(filename.c_str());
    if (!infile.is_open()) return false;

    string line;
    bool header = true;
    vector<vector<double>> rows;
    while (getline(infile, line)) {
        if (header) { header = false; continue; } // skip header
        // crude split to check columns
        vector<string> cols;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) cols.push_back(cell);
        if (cols.size() < 5) continue; // refuse small/invalid rows

        // parse with Point parser to be robust with non-only-commas splits
        Point tmp((int)rows.size()+1, line);
        int d = tmp.getDimensions();
        if (rows.empty()) {
            D = d;
        } else if (D != d) {
            // irregular row, skip
            continue;
        }
        vector<double> vec(d);
        for (int j = 0; j < d; j++) vec[j] = tmp.getVal(j);
        rows.push_back(std::move(vec));
    }
    infile.close();
    N = (int)rows.size();
    if (N == 0) return false;
    flat.resize((size_t)N * (size_t)D);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) flat[(size_t)i*D + j] = rows[i][j];
    }
    return true;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc != 3) {
        if (rank == 0) {
            cout << "Usage: ./kmeans_mpi_bcast <INPUT-FILE> <OUT-DIR>\n";
        }
        MPI_Finalize();
        return 1;
    }
    string filename = argv[1];
    string outdir   = argv[2];

    double t0 = 0.0;
    if (rank == 0) t0 = omp_get_wtime();

    // ---------------------- MPI data distribution ----------------------
    int N = 0, D = 0;
    vector<double> flat;

    if (rank == 0) {
        bool ok = read_csv_rank0(filename, flat, N, D);
        if (!ok) {
            cout << "Error: failed to read/parse input file '" << filename << "'.\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    // Broadcast N and D
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate receive buffer on non-root
    if (rank != 0) flat.resize((size_t)N * (size_t)D);

    // Broadcast raw data
    if (N > 0 && D > 0) {
        MPI_Bcast(flat.data(), (int)flat.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        if (rank == 0) cout << "Error: empty dataset after parsing.\n";
        MPI_Finalize();
        return 1;
    }

    // Rebuild Points locally
    vector<Point> all_points;
    all_points.reserve(N);
    for (int i = 0; i < N; i++) {
        all_points.emplace_back(i+1, flat.data() + (size_t)i*D, D);
    }

    if (rank == 0) {
        cout << "\nData fetched successfully! Total points: " << all_points.size()
             << " | dimensions: " << D << "\n";
    }

    // -------------------------- Elbow params ---------------------------
    const int K_MIN = 2;
    const int K_MAX = 20;
    const int IT_MAX = 200;

    if ((int)all_points.size() < K_MAX && rank == 0) {
        cout << "Warning: Number of points is less than K_MAX. Reducing effective K range.\n";
    }

    // Local WCSS results
    vector<pair<int,double>> local_wcss;

    if (rank == 0) {
        cout << "\n--- Running Elbow Method (K=" << K_MIN << " to " << K_MAX << ") ---\n\n";
        cout << "K | WCSS (Within-Cluster Sum of Squares)\n";
        cout << "------------------------------------------\n";
    }

    // Cyclic K distribution among ranks
    for (int K = K_MIN + rank; K <= K_MAX && K < (int)all_points.size(); K += world) {
        vector<Point> pts = all_points; // copy to reset assignments
        KMeans km(K, IT_MAX, outdir);
        km.run(pts);
        double wcss = km.calculateWCSS();
        local_wcss.emplace_back(K, wcss);
    }

    // ------------------------ Gather to rank 0 -------------------------
    if (rank == 0) {
        vector<pair<int,double>> global = local_wcss;
        for (int p = 1; p < world; p++) {
            int rcount = 0;
            MPI_Recv(&rcount, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rcount > 0) {
                vector<int> ks(rcount);
                vector<double> vs(rcount);
                MPI_Recv(ks.data(), rcount, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(vs.data(), rcount, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < rcount; i++) global.emplace_back(ks[i], vs[i]);
            }
        }
        sort(global.begin(), global.end());
        for (auto &kv : global) {
            cout << fixed << setprecision(4) << kv.first << " | " << kv.second << "\n";
        }
        cout << "\n=======================================================\n"
             << "RESULTADOS DO MÉTODO DO COTOVELO:\n"
             << "O 'K' ideal (o cotovelo) é o ponto onde o valor do WCSS para de diminuir rapidamente.\n";

        double t1 = omp_get_wtime();
        printf("\nTempo Total de Execução: %f segundos (para todas as iterações de K)\n", t1 - t0);
    } else {
        int sendc = (int)local_wcss.size();
        MPI_Send(&sendc, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (sendc > 0) {
            vector<int> ks(sendc);
            vector<double> vs(sendc);
            for (int i = 0; i < sendc; i++) {
                ks[i] = local_wcss[i].first;
                vs[i] = local_wcss[i].second;
            }
            MPI_Send(ks.data(), sendc, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(vs.data(), sendc, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
