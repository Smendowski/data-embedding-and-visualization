## 1. Introduction
<p style="text-align:justify">
Visualization makes it easier to understand and notice dependencies in the high dimensionality data
that are not trivial to capture and perceive. It is an inseparable, far-reaching, and effectual concept
of data analysis or its initial recognition, but also an autonomous tool and dextrous field of machine
learning. Visualization allows checking whether there are groups of similar observations forming clusters
and finally gain more priceless intuition and understanding about data. In the case of multi and highdimensional
ones, it is necessary to reduce their dimensions to at most three. The relationships in
data are often non-linear, which rules out methods like PCA regarding separation quality.
Therefore, it is required to use Manifold Learning techniques to discover the surface (manifold) on which
the data is distracted and make reasonable projections into a space with the desired dimensionality. This project aims to analyze and visualize the MINST, 20 News Groups, and RCV Reuters datasets
using methods such as t-SNE, UMAP, ISOMAP, PaCMAP and IVHD. Therefore, the particular motivation
is to show the concept of high-dimensional data visualization, assess multiple data embedding
techniques, and highlight potential comparative criteria of data separation quality.
</p>

## 2. VISKIT
### 1. Configuration and Setup
[Viskit Repository and README](https://gitlab.com/bminch/viskit)
```bash
git clone https://gitlab.com/bminch/viskit.git
docker build -t viskit -f Dockerfile .
docker run -it viskit /bin/bash
```

### 2. Graphs
Graphs are required by VisKit. For this project, they can be downloaded either manually or automatically.
```bash
source /utils/download_graphs.sh
```
Graphs location on Google Drive: <br>
[mnist_cosine.bin](https://drive.google.com/file/d/1vhY_dvn30s_muTN7-vzqKzKSWpRb-YWW/view?usp=sharing) <br>
[mnist_euclidean.bin](https://drive.google.com/file/d/1SYScDuxFx9-kYpFHljloVuZFZzR7OmBh/view?usp=sharing) <br>
[reuters_cosine.bin](https://drive.google.com/file/d/1QhLo11NKZ_DpFLNcF33e8sew0K7d7UI4/view?usp=sharing) <br>
[reuters_euclidean.bin](https://drive.google.com/file/d/1k4WH6piQmHP3c8DixlbmVF99wCF4ysNo/view?usp=sharing) <br>
[tng_cosine.bin](https://drive.google.com/file/d/1j3O0EIZE3A-eNNy08gFGUK5h86djlOJ-/view?usp=sharing) <br>
[tng_euclidean.bin](https://drive.google.com/file/d/1KDgp8hnX8hTN4M9APWzOVOWm2qi-aVDl/view?usp=sharing) <br>

### 3. Usage documentation
Provide dataset (without labels; <b>path_to_dataset_file</b>), labels (<b>path_to_labels_file</b>) as separate csv files and graph file (<b>{path_to_graph_file}</b>). Visualization text file will be saved to specified path (<b>path_to_visualization</b>).
```bash
cd /opt/viskit/viskit_offline
./viskit_offline {path_to_dataset_file} {path_to_labels_file} {path_to_graph_file} {path_to_visualization} 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline {path_to_dataset_file} {path_to_labels_file} {path_to_graph_file} {path_to_visualization}
```

### 4. Usage examples
```bash
cd /opt/viskit/viskit_offline
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt
```

## 3. Metrics
Metrics are used to asses and compare quality of dimensionality reduction techniques. Two major aspects are worth to include during assesment - the local and global quality of separation.

**Implemented Metrics:**
1. *Distance matrix-based metric*
2. *Distance matrix-based metric with KMeans optimization*
3. *KMeans extension of distance matrix based metric*
4. *Thrustworthiness-based metric*
5. *Spearman correlation-based metric*
6. *KNN Gain & DR Quality*
7. *Sheppard Diagram*
8. *Co-ranking matrix-based metric*

## 4. [Appendix] Introduction to Dimenstionality Reduction
Jupyter notebooks that covers basic and advanced issues regarding the visualization of large data sets and Dimensionality Reduction
1. [Principal Component Analysis](https://github.com/Smendowski/data-embedding-and-visualization/blob/main/!Introduction%20to%20DR/%5B1%5D%20Principal%20Component%20Analysis.ipynb)
2. [Roulade projections using t-SNE and MDS](https://github.com/Smendowski/data-embedding-and-visualization/blob/main/!Introduction%20to%20DR/%5B2%5D%20MDS%20and%20t-SNE%20projections%20of%20custom%20roulade%20data.ipynb)
3. [f-MNIST and MNIST visualizations using t-SNE, UMAP and LargeVis](https://github.com/Smendowski/data-embedding-and-visualization/blob/main/!Introduction%20to%20DR/%5B3%5D%20f-MNIST%20and%20MNIST%20projections%20using%20t-SNE%20UMAP%20%26%20LargeVis.ipynb)
4. [Neural Networks hidden layers activations embedding](https://github.com/Smendowski/data-embedding-and-visualization/blob/main/!Introduction%20to%20DR/%5B4%5D%20Neural%20Network%20hidden%20layers%20activation%20embedding.ipynb)

## 5. [Documentation](https://github.com/Smendowski/data-embedding-and-visualization/blob/main/!Documentation/Large%20Datasets%20Embedding%20and%20Visualization%20-%20documentation.pdf)

## 6. Authors
Mateusz Smendowski & Michał Grela