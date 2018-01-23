#include <ctime>
#include "graph.h"

void readGraph(Graph &G, int argc, char **argv) {
    int n;
    int m;

    //If no arguments then read graph from stdin
    bool fromStdin = argc <= 2;
    if (fromStdin) {
        scanf("%d %d", &n, &m);
    } else {
        srand(12345);
        n = atoi(argv[2]);
        m = atoi(argv[3]);
    }

    std::vector<std::vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        if (fromStdin) {
            scanf("%d %d", &u, &v);
            adjecancyLists[u].push_back(v);
        } else {
            u = rand() % n;
            v = rand() % n;
            adjecancyLists[u].push_back(v);
            adjecancyLists[v].push_back(u);
        }
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
