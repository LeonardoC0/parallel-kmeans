/* TEMPO DE EXECUÇÕES
Tempo Total de Execução: 22.511 segundos (para todas as iterações de K)
*/
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream> 
#include <time.h> // Para medição de tempo
#include <iomanip> // Para formatação de saída

#ifdef _OPENMP
#include <omp.h>
#else
// Simulação de omp_get_wtime se OpenMP não estiver disponível
double omp_get_wtime() { return (double)clock() / CLOCKS_PER_SEC; }
#endif

using namespace std;

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
        
        // O separador é a vírgula, mas o lineToVec original não usava vírgula
        // Ele fazia um parse complexo de string. Manter a lógica original
        // para não quebrar a funcionalidade de leitura interna do Point.
        for (int i = 0; i < (int)line.length(); i++)
        {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')
            {
                tmp += line[i];
            }
            else if (tmp.length() > 0)
            {
                try {
                    values.push_back(stod(tmp));
                } catch (...) {
                    // Ignora conversão inválida (como strings vazias)
                }
                tmp = "";
            }
        }
        if (tmp.length() > 0)
        {
            try {
                values.push_back(stod(tmp));
            } catch (...) {
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
    Cluster(int clusterId, const Point& centroid_p)
    {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid_p.getDimensions(); i++)
        {
            this->centroid.push_back(centroid_p.getVal(i));
        }
        // Não adicionamos o ponto aqui, pois o ponto inicial é apenas o centróide.
        // A adição real ocorre no loop de iteração do K-Means.
    }

    void addPoint(const Point& p)
    {
        // A atribuição do cluster ID para 'p' deve ser feita no KMeans::run
        // antes de chamar addPoint.
        points.push_back(p);
    }

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
        for (int i = 0; i < K; i++)
        {
            clusters[i].removeAllPoints();
        }
    }

    // Calcula a distância euclidiana ao quadrado
    double calculateSquaredDistance(const Point& point, const Cluster& cluster)
    {
        double sum = 0.0;
        for (int i = 0; i < dimensions; i++)
        {
            double diff = cluster.getCentroidByPos(i) - point.getVal(i);
            sum += diff * diff;
        }
        return sum;
    }

    int getNearestClusterId(const Point& point)
    {
        // Usamos a distância euclidiana ao quadrado para evitar a raiz quadrada (otimização)
        double min_dist_sq = calculateSquaredDistance(point, clusters[0]);
        int NearestClusterId = clusters[0].getId();

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

        for (const auto& cluster : clusters)
        {
            double cluster_wcss = 0.0;
            for (int p = 0; p < cluster.getSize(); p++)
            {
                // Calcula a distância euclidiana ao quadrado do ponto ao seu centróide
                const Point& point = cluster.getPoint(p);
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
        if (total_points == 0) return;
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
                if (all_points[i].getCluster() > 0) {
                    clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
                }
            }

            // 3. Recalcular o centróide de cada cluster
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
        if (K == 10) {
            ofstream pointsFile;
            pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
            for (int i = 0; i < total_points; i++) {
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

    // Agora precisamos apenas do arquivo de entrada e do diretório de saída
    if (argc != 3)
    {
        cout << "Error: command-line argument count mismatch. \n Usage: ./elbow_kmeans <INPUT-FILE> <OUT-DIR>" << endl;
        return 1;
    }

    string output_dir = argv[2];
    string filename = argv[1];
    
    // --- Leitura e Preparação dos Dados (Inalterado) ---
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
        if (header) { 
            header = false; 
            continue; // pular cabeçalho
        }

        vector<string> cols;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ','))
            cols.push_back(cell);

        if (cols.size() < 5) continue; 


        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }

    infile.close();
    cout << "\nData fetched successfully! Total points: " << all_points.size() << endl;

    if (all_points.empty()) {
        cout << "Error: Input file is empty or data could not be parsed." << endl;
        return 1;
    }

    // --- Implementação do Método do Cotovelo ---
    const int K_MIN = 2; 
    const int K_MAX = 20; 
    const int IT_MAX = 200; // Máximo de iterações por execução do K-Means

    if ((int)all_points.size() < K_MAX)
    {
        cout << "Warning: Number of points is less than K_MAX. Reducing K_MAX to " << all_points.size() - 1 << endl;
        // Se K_MAX for maior que o número de pontos, ajustamos K_MAX para ser seguro
        // (pelo menos 1 ponto por cluster é necessário, mas K > N é inútil)
        // Usaremos o menor valor entre o K_MAX definido e o número de pontos - 1.
        // Se K_MIN for maior que este valor, o loop será ignorado.
        
    }
    
    // É crucial ter uma cópia original para resetar o estado dos pontos a cada iteração de K
    const vector<Point> original_points = all_points; 
    vector<pair<int, double>> wcss_results; 

    cout << "\n--- Running Elbow Method (K=" << K_MIN << " to K=" << K_MAX << ") ---\n" << endl;
    cout << "K | WCSS (Within-Cluster Sum of Squares)\n";
    cout << "------------------------------------------\n";

    for (int K = K_MIN; K <= K_MAX && K < (int)original_points.size(); ++K)
    {
        // 1. Resetar pontos para novo run: usamos a cópia da cópia
        vector<Point> current_points = original_points;
        
        // 2. Executar K-Means
        KMeans kmeans(K, IT_MAX, output_dir);
        kmeans.run(current_points); 

        // 3. Calcular WCSS
        double wcss = kmeans.calculateWCSS(); 
        wcss_results.push_back({K, wcss});

        cout << fixed << setprecision(4) << K << " | " << wcss << endl;
    }
    
    // --- Análise dos Resultados ---
    cout << "\n=======================================================\n";
    cout << "RESULTADOS DO MÉTODO DO COTOVELO:\n";
    cout << "O 'K' ideal (o cotovelo) é o ponto onde o valor do WCSS para de diminuir rapidamente.\n";
    cout << "Para visualizar o cotovelo, os dados de K e WCSS devem ser plotados em um gráfico.\n";

    end_time = omp_get_wtime();
    printf("\nTempo Total de Execução: %f segundos (para todas as iterações de K)\n", end_time - start_time);

    return 0;
}
