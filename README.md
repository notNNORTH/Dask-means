# On Simplifying Large-Scale Spatial Vectors: Fast, Memory-Efficient, and Cost-Predictable k-means

## Introduction

***Dask-means*** is a fast, memory-efficient, and cost-predictable <u>**da**</u>taset <u>**s**</u>implification **<u>k</u>-means** algorithm for large-scale spatial vectors. This repo holds the source code and scripts for reproducing the key experiments of our paper: *On Simplifying Large-Scale Spatial Vectors: Fast, Memory-Efficient, and Cost-Predictable k-means*.

## Datasets

|    Datasets    | T-drive | Proto | Argo-AVL | Argo-PC | 3D-RD | Shapenet |
| :------------: | :-----: | :---: | :------: | :-----: | :---: | :------: |
| Dimensionality |    2    |   2   |    2     |    3    |   3   |    3     |
| Dataset Scale  |   11M   | 1.27M |   177M   |  383M   |  4M   |   100M   |

The experiments used low-dimensional point cloud datasets, and all datasets can be acquired via the web we list.

- T-drive: [T-Drive trajectory data sample - Microsoft Research](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
- Proto: [Porto taxi trajectories (figshare.com)](https://figshare.com/articles/dataset/Porto_taxi_trajectories/12302165?file=22677902)
- Argo-AVL: [[Argoverse 2](https://www.argoverse.org/av2.html)](https://www.argoverse.org/index.html)
- Argo-PC: [Argoverse 2](https://www.argoverse.org/av2.html)
- 3D-RD: [3D-spatial-network | Machine Learning Data (networkrepository.com)](https://networkrepository.com/3D-spatial-network.php)
- Shapenet: [ShapeNet](https://shapenet.org/)

## How to Run Dask-means

### Code with comparative algorithms:

- You can run the code in `./src/edu/nyu/unik/expriments/kmeansEfficiency.java` to obtain the runtime of each comparative algorithm.
- You can go to the file `./src/au/edu/rmit/trajectory/clustering/kmeans/keamsAlgorithm.java`, and select your comparative algorithms in function `test_DasKmeans`.

### Runtime Adjustment:

- You can find all the relevant code for runtime adjustments in `./refinement/`.
- `kmeans-time-total.csv` and `kmeans-time-series.csv` are the training datasets that represent the overall runtime and the time of each iteration, respectively.
- The code for runtime prediction is in the file `main.py`.
- The code for runtime adjustment is in the file `GaussinProcess.py`.
