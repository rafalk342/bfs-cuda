#include <cstdio>
#include <cuda.h>
#include <string>

#include "graph.h"
#include "bfsCPU.h"

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

void checkError(CUresult error, std::string msg) {
    if (error != CUDA_SUCCESS) {
        printf("%s: %d\n", msg.c_str(), error);
        exit(1);
    }
}

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

CUfunction cuSimpleBfs;
CUfunction cuQueueBfs;
CUfunction cuNextLayer;
CUfunction cuCountDegrees;
CUfunction cuScanDegrees;
CUfunction cuAssignVerticesNextQueue;

CUdeviceptr d_adjacencyList;
CUdeviceptr d_edgesOffset;
CUdeviceptr d_edgesSize;
CUdeviceptr d_distance;
CUdeviceptr d_parent;
CUdeviceptr d_currentQueue;
CUdeviceptr d_nextQueue;
CUdeviceptr d_degrees;
int *incrDegrees;

void initCuda(Graph &G) {
    //initialize CUDA
    cuInit(0);
    checkError(cuDeviceGet(&cuDevice, 1), "cannot get device 0");
    checkError(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    checkError(cuModuleLoad(&cuModule, "bfsCUDA.ptx"), "cannot load module");
    checkError(cuModuleGetFunction(&cuSimpleBfs, cuModule, "simpleBfs"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuQueueBfs, cuModule, "queueBfs"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuNextLayer, cuModule, "nextLayer"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuCountDegrees, cuModule, "countDegrees"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuScanDegrees, cuModule, "scanDegrees"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuAssignVerticesNextQueue, cuModule, "assignVerticesNextQueue"),
               "cannot get kernel handle");

    //copy memory to device
    checkError(cuMemAlloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
    checkError(cuMemAlloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
    checkError(cuMemAlloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
    checkError(cuMemAlloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
    checkError(cuMemAlloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
    checkError(cuMemAlloc(&d_currentQueue, G.numVertices * sizeof(int)), "cannot allocate d_currentQueue");
    checkError(cuMemAlloc(&d_nextQueue, G.numVertices * sizeof(int)), "cannot allocate d_nextQueue");
    checkError(cuMemAlloc(&d_degrees, G.numVertices * sizeof(int)), "cannot allocate d_degrees");
    checkError(cuMemAllocHost((void **) &incrDegrees, sizeof(int) * G.numVertices), "cannot allocate memory");

    checkError(cuMemcpyHtoD(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int)),
               "cannot copy to d_adjacencyList");
    checkError(cuMemcpyHtoD(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_edgesOffset");
    checkError(cuMemcpyHtoD(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_edgesSize");


}

void finalizeCuda() {
    //free memory
    checkError(cuMemFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
    checkError(cuMemFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
    checkError(cuMemFree(d_edgesSize), "cannot free memory for d_edgesSize");
    checkError(cuMemFree(d_distance), "cannot free memory for d_distance");
    checkError(cuMemFree(d_parent), "cannot free memory for d_parent");
    checkError(cuMemFree(d_currentQueue), "cannot free memory for d_parent");
    checkError(cuMemFree(d_nextQueue), "cannot free memory for d_parent");
    checkError(cuMemFreeHost(incrDegrees), "cannot free memory for incrDegrees");
}

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
    for (int i = 0; i < G.numVertices; i++) {
        if (distance[i] != expectedDistance[i]) {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
               "cannot copy to d)distance");
    checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_parent");

    int firstElementQueue = startVertex;
    cuMemcpyHtoD(d_currentQueue, &firstElementQueue, sizeof(int));
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //copy memory from device
    checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
               "cannot copy d_distance to host");
    checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");

}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *changed;
    checkError(cuMemAllocHost((void **) &changed, sizeof(int)), "cannot allocate changed");

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;
        void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
                        &changed};
        checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
                                  1024, 1, 1, 0, 0, args, 0),
                   "cannot run kernel simpleBfs");
        cuCtxSynchronize();
        level++;
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
                     std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *nextQueueSize;
    checkError(cuMemAllocHost((void **) &nextQueueSize, sizeof(int)), "cannot allocate nextQueueSize");

    //launch kernel
    printf("Starting queue parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    *nextQueueSize = 0;
    int level = 0;
    while (queueSize) {
        void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
                        &nextQueueSize, &d_currentQueue, &d_nextQueue};
        checkError(cuLaunchKernel(cuQueueBfs, queueSize / 1024 + 1, 1, 1,
                                  1024, 1, 1, 0, 0, args, 0),
                   "cannot run kernel queueBfs");
        cuCtxSynchronize();
        level++;
        queueSize = *nextQueueSize;
        *nextQueueSize = 0;
        std::swap(d_currentQueue, d_nextQueue);
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

void nextLayer(int level, int queueSize) {
    void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
                    &d_currentQueue};
    checkError(cuLaunchKernel(cuNextLayer, queueSize / 1024 + 1, 1, 1,
                              1024, 1, 1, 0, 0, args, 0),
               "cannot run kernel cuNextLayer");
    cuCtxSynchronize();
}

void countDegrees(int level, int queueSize) {
    void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize,
                    &d_currentQueue, &d_degrees};
    checkError(cuLaunchKernel(cuCountDegrees, queueSize / 1024 + 1, 1, 1,
                              1024, 1, 1, 0, 0, args, 0),
               "cannot run kernel cuNextLayer");
    cuCtxSynchronize();
}

void scanDegrees(int queueSize) {
    //run kernel so every block in d_currentQueue has prefix sums calculated
    void *args[] = {&queueSize, &d_degrees, &incrDegrees};
    checkError(cuLaunchKernel(cuScanDegrees, queueSize / 1024 + 1, 1, 1,
                              1024, 1, 1, 0, 0, args, 0), "cannot run kernel scanDegrees");
    cuCtxSynchronize();

    //count prefix sums on CPU for ends of blocks exclusive
    //already written previous block sum
    incrDegrees[0] = 0;
    for (int i = 1024; i < queueSize + 1024; i += 1024) {
        incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
    }
}

void assignVerticesNextQueue(int queueSize, int nextQueueSize) {
    void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize, &d_currentQueue,
                    &d_nextQueue, &d_degrees, &incrDegrees, &nextQueueSize};
    checkError(cuLaunchKernel(cuAssignVerticesNextQueue, queueSize / 1024 + 1, 1, 1,
                              1024, 1, 1, 0, 0, args, 0),
               "cannot run kernel assignVerticesNextQueue");
    cuCtxSynchronize();
}

void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
                    std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    //launch kernel
    printf("Starting scan parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    while (queueSize) {
        // next layer phase
        nextLayer(level, queueSize);
        // counting degrees phase
        countDegrees(level, queueSize);
        // doing scan on degrees
        scanDegrees(queueSize);
        nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
        // assigning vertices to nextQueue
        assignVerticesNextQueue(queueSize, nextQueueSize);

        level++;
        queueSize = nextQueueSize;
        std::swap(d_currentQueue, d_nextQueue);
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

int main(int argc, char **argv) {

    // read graph from standard input
    Graph G;
    int startVertex = atoi(argv[1]);
    readGraph(G, argc, argv);

    printf("Number of vertices %d\n", G.numVertices);
    printf("Number of edges %d\n\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    //run CPU sequential bfs
    runCpu(startVertex, G, distance, parent, visited);

    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);

    initCuda(G);
    //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);

    //run CUDA queue parallel bfs
    runCudaQueueBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);

    //run CUDA scan parallel bfs
    runCudaScanBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);

    finalizeCuda();
    return 0;
}