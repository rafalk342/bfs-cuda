#ifndef BFS_CUDA_GRAPH_H
#define BFS_CUDA_GRAPH_H

#include <vector>
#include <cstdio>
#include <cstdlib>

struct Graph {
    std::vector<int> adjacencyList; // all edges
    std::vector<int> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    int numVertices = 0;
    int numEdges = 0;
};

void readGraph(Graph &G, int argc, char **argv);

#endif //BFS_CUDA_GRAPH_H
