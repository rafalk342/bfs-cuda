#include "graph.h"

void readGraph(Graph &G, int n, int m) {
    srand(12345);


    std::vector<std::vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
        int u = rand() % n;
        int v = rand() % n;
        adjecancyLists[u].push_back(v);
        adjecancyLists[v].push_back(u);
    }

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }
    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
}
