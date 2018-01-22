

#include <device_launch_parameters.h>

extern "C" {

__global__
void simpleBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
               int *d_edgesSize, int *d_distance, int *d_parent, int *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if(thid < N && d_distance[thid] == level){
        int u = thid;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[u] + 1 < d_distance[v]) {
                d_distance[v] = d_distance[u] + 1;
                d_parent[v] = u;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}

}
