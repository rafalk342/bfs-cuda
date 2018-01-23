# BFS CUDA

## Description: 
Different approaches for implementation of bfs on GPU with CUDA Driver API.  

## Algorithms:
- O(V + E) sequential bfs
- O(V^2 + E) parallel simple bfs
- O(V + E) parallel queue with atomic operations bfs (slow)
- O(V + E) parallel queue with scan bfs

## Usage:
```
To build the project run:
make

To run algorithms on random generated graphs:
./main <start vertex> <number of vertices> <number of edges>

To run algorithm on graphs from standard input:
./main <start vertex> < input
Input should be in the form:
<number of vertices> <number of edges>
<end of edge1> <end of edge1>
<end of edge2> <end of edge2>
...

```

## Links:
Accelerating large graph algorithms on the GPU using CUDA, Pawan Harish and P. J. Narayanan
https://pdfs.semanticscholar.org/4c77/e5650e2328390995f3219ec44a4efd803b84.pdf

Scalable GPU Graph Traversal, Duane Merrill, Michael Garland, Andrew Grimshaw
http://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf

There is also an O(V + E) algorithm described below that uses hierarchical queues and works efficiently 
with shared memory but it needs to convert graph into a near regular-graph before running the kernels.  

An Effective GPU Implementation of Breadth-First Search, Lijuan Luo, Martin Wong, Wen-mei Hwu  
http://impact.crhc.illinois.edu/shared/papers/effective2010.pdf