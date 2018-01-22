
#include <chrono>
#include <cstdio>

#include "graph.h"
#include "bfsCPU.h"

int main() {
    // read graph from standard input
    Graph G;
    readGraph(G);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);


//    printf("Number of vertices: %d \n", G.numVertices);
//    printf("Number of edges: %d \n", G.numEdges);
//
//    for (int i = 0; i < G.numVertices; i++) {
//        printf("Edges for %d :", i);
//        for (int j = G.edgesOffset[i]; j < G.edgesOffset[i] + G.edgesSize[i]; j++) {
//            printf("%d ", G.adjacencyList[j]);
//        }
//        printf("\n");
//    }

    //run CPU sequential bfs
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(0, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);

//    //Printing out results
//    for(auto &a:distance){
//        printf("%d ", a);
//    }
//    printf("\n");
//    for(auto &a:parent){
//        printf("%d ", a);
//    }
    //run CUDA parallel bfs
    return 0;
}