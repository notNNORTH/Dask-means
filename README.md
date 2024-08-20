# On Simplifying Large-Scale Spatial Vectors: Fast, Memory-Efficient, and Cost-Predictable k-means

## Introduction

***Dask-means*** is a fast, memory-efficient, and cost-predictable <u>**da**</u>taset <u>**s**</u>implification **<u>$k$</u>-means** algorithm for large-scale spatial vectors. This repo holds the source code and scripts for reproducing the key experiments of our paper: *On Simplifying Large-Scale Spatial Vectors: Fast, Memory-Efficient, and Cost-Predictable $k$-means*.

## Datasets

The datasets we use are all low-dimensional (2-3 dimensional) spatial vectors, with brief information as shown in the list below:

|    Datasets    | T-drive | Proto | Argo-AVL | Argo-PC | 3D-RD | Shapenet |
| :------------: | :-----: | :---: | :------: | :-----: | :---: | :------: |
| Dimensionality |    2    |   2   |    2     |    3    |   3   |    3     |
| Dataset Scale  |   11M   | 1.27M |   177M   |  383M   |  4M   |   100M   |

Furthermore, the specific information and download links for each dataset are as follows:

**T-drive**:

- The T-Drive trajectory dataset contains GPS trajectories of 10,357 taxis in Beijing from February 2 to February 8, 2008.
- Link: [T-Drive trajectory dataset](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)

**Proto**:

- A CSV dataset containing taxi trajectories recorded over one year (from 2013/07/01 to 2014/06/30) in the city of Porto, in Portugal.
- Link: [Porto taxi trajectories](https://figshare.com/articles/dataset/Porto_taxi_trajectories/12302165?file=22677902)

**Argo-AVL**: 

- Trajectory dataset from Argoverse, recording the trajectory data of autonomous vehicles.
- Link: [Argoverse 2: Trajectory](https://www.argoverse.org/av2.html)

**Argo-PC**: 

- Point cloud dataset from Argoverse, representing lidar-detected objects surrounding vehicles.
- Link: [The 3D Lidar Object Detection and Tracking Challenge of Apolloscape Dataset](https://github.com/ApolloScapeAuto/dataset-api/tree/master/3d_detection_tracking)

**3D-RD**: 

- The 3D-spatial-network dataset is a machine learning dataset that contains 3D road network information of the North Jutland region in Denmark.
- Link: [3D-spatial-network | Machine Learning Data](https://networkrepository.com/3D-spatial-network.php)

**Shapenet**:

- Shapenet is a widely used point cloud dataset for 3D shape understanding and analysis.
- Link: [Shapenet](https://shapenet.org/)


## Comparison Algorithm

#### [Least squares quantization in PCM](https://hal.science/hal-04614938/document)

- The most widely used $k$-means algorithm.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [Using the Triangle Inequality to Accelerate k-Means](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)

- By maintaining the upper and lower bounds of all spatial vectors to all cluster centroids and using the triangle inequality to accelerate $k$-means.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [Making k-means even faster](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12)

- Using the triangle inequality, but for each spatial vector, only maintaining the minimum lower bound to any cluster centroid.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [Accelerated k-means with adaptive distance bounds](http://opt.kyb.tuebingen.mpg.de/papers/opt2012_paper_13.pdf)

- Using the triangle inequality, but for each spatial vector, only maintaining $1<b<k$ lower bound to any cluster centroid.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [Accelerating Lloyd’s algorithm for k-means clustering](https://cs.baylor.edu/~hamerly/papers/2014_pca_chapter_hamerly_drake.pdf)

- Accelerate the $k$-means algorithm using Heap-structured bounds.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [Yinyang K-Means: A Drop-In Replacement of the Classic K-Means with Consistent Speedup](https://proceedings.mlr.press/v37/ding15.pdf)

- Group the clusters and maintain bounds for each group, implementing a pruning method with multiple filters.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [A Fast Adaptive k-means with No Bounds](https://par.nsf.gov/servlets/purl/10286756)

- Organize spatial vectors using a hyperspherical structure to avoid boundary maintenance.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

#### [A Dual-Tree Algorithm for Fast k-means Clustering with Large k](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.34)

- Organize spatial vectors using a dual-tree structure to accelerate the execution of the $k$-means algorithm.
- The code is in the `test_DasKmeans()` function under the `./src/au/edu/rmit/trajectory/clustering/kmeans/kmeansAlgorithm.java` path.

## How to Run Dask-means

### Code with comparative algorithms

- You can run the code in `./src/edu/nyu/unik/expriments/kmeansEfficiency.java` to obtain the runtime of each comparative algorithm.
- You can go to the file `./src/au/edu/rmit/trajectory/clustering/kmeans/keamsAlgorithm.java`, and select your comparative algorithms in function `test_DasKmeans()`.

**Parameter configuration:**

There are seven parameters in the file  `./src/edu/nyu/unik/expriments/kmeansEfficiency.java`. Remember to set the parameters to the correct format before running.

```
paras[0] is the path to spatial vector dataset
paras[1] is the number of clusters (k)
paras[2] is the number of spatial vectors (|D|)
paras[3] is set as the default parameter for "a"
paras[4] is the setting for the output file name
paras[5] is the initial data dimension of the dataset
paras[6] is the terminating data dimension of the dataset.
```

### Cost Model

#### Memory Prediction

The core code for calculating memory overhead is in the function `getAllMemory()` in file `./src/au/edu/rmit/trajectory/clustering/kmeans/keamsAlgorithm.java`. This function will be called during the execution of each comparison algorithm in `test_DasKmeans()`.

Moreover, you can also manually adjust the capacity of leaf nodes in function `experiments()` of file `./src/au/edu/rmit/trajectory/clustering/kmeans/keamsAlgorithm.java`.

#### Runtime Prediction

- You can find all the relevant code for runtime adjustments in `./refinement/`.
- `kmeans-time-total.csv` and `kmeans-time-series.csv` are the training datasets that represent the overall runtime and the time of each iteration, respectively.
- The code for runtime prediction is in the file `main.py`.
- The code for runtime adjustment is in the file `GaussinProcess.py`.

