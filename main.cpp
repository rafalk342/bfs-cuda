#include <iostream>
#include <vector>
#include <queue>
#include <chrono>

struct Node {
    int distance;
    int parent;
    int visited = false;
};

void bfsVertexCentric(int s, std::vector<Node> &Graph, std::vector<std::vector<int>> &GraphAdjacency) {
    Graph[s].distance = 0;
    Graph[s].parent = s;
    std::queue<int> Q;
    Q.push(s);
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
        for (auto &v:GraphAdjacency[u]) {
            if (!Graph[v].visited) {
                Graph[v].visited = true;
                Graph[v].distance = Graph[u].distance + 1;
                Graph[v].parent = u;
                Q.push(v);
            }
        }
    }
}

void bfsEdgeCentric(int s, std::vector<Node> &Graph, std::vector<std::vector<bool>> &GraphMatrix) {
    Graph[s].distance = 0;
    Graph[s].parent = s;
    std::queue<int> Q;
    Q.push(s);
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
        for (int v = 0; v < Graph.size(); v++) {
            if (GraphMatrix[u][v] && !Graph[v].visited) {
                Graph[v].visited = true;
                Graph[v].distance = Graph[u].distance + 1;
                Graph[v].parent = u;
                Q.push(v);
            }
        }
    }
}

int main() {
    //start time
    auto start = std::chrono::steady_clock::now();
    // read some graph
    int n, m;
    std::cin >> n >> m;
    std::vector<Node> Graph(n);
    std::vector<std::vector<int> > GraphAdjacency(n);
    std::vector<std::vector<bool> > GraphMatrix(n, std::vector<bool>(n, false));

    for (int i = 0; i < m; i++) {
        int u, v;
        std::cin >> u >> v;
        GraphAdjacency[u].push_back(v);
        GraphAdjacency[v].push_back(u);
//        GraphMatrix[u][v] = true;
//        GraphMatrix[v][u] = true;
    }
    // run bfsVertexCentric on it
    bfsVertexCentric(0, Graph, GraphAdjacency);
//    bfsEdgeCentric(0, Graph, GraphMatrix);
    // keep information about execution time

    // output acquired results
    //output distance
    for (auto &u:Graph) {
        std::cout << u.distance << ' ';
    }
    std::cout << std::endl;
    //output parents
    for (auto &u:Graph) {
        std::cout << u.parent << ' ';
    }
    std::cout << std::endl;

    //end time
    auto end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time in milliseconds : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    return 0;
}