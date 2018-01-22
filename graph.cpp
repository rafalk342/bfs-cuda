#include "graph.h"

void readGraph(Graph &G) {
    int n, m;
    scanf("%d %d", &n, &m);

    G.numVertices = n;
    G.numEdges = m;

    std::vector<std::vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adjecancyLists[u].push_back(v);
    }

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }
}
