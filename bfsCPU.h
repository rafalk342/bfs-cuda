#ifndef BFS_CUDA_BFSCPU_H
#define BFS_CUDA_BFSCPU_H

#include <chrono>
#include <queue>
#include "graph.h"

void bfsCPU(int start, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited);

#endif //BFS_CUDA_BFSCPU_H
