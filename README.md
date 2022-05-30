```bash
git clone https://gitlab.com/bminch/viskit.git
docker build -t viskit -f Dockerfile .
docker run -it viskit /bin/bash
```

## Download graphs
Graphs *.bin files required by VisKit can be downloaded either manually or automatically.
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


## Viskit usage
You need to provide dataset (without labels; <b>path_to_dataset_file</b>), labels (<b>path_to_labels_file</b>) as separate csv files and graph file (<b>{path_to_graph_file}</b>). Visualization text file will be saved to specified path (<b>path_to_visualization</b>).
```bash
cd /opt/viskit/viskit_offline
./viskit_offline {path_to_dataset_file} {path_to_labels_file} {path_to_graph_file} {path_to_visualization} 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline {path_to_dataset_file} {path_to_labels_file} {path_to_graph_file} {path_to_visualization}
```

### Examples
```bash
cd /opt/viskit/viskit_offline
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt
```

## Metrics
Metrics are used to asses and compare quality of dimensionality reduction techniques. Two major aspects are worth to evaluate - the local and global quality of separation. 

Currently supported metrics:
1. `Distance matrix based metric`

```python
from sklearn.manifold import TSNE
from metrics.distance_matrix_based_metric import DistanceMatrixBasedMetric


df_mnist = pd.read_csv(
    './datasets_with_labels/mnist.csv', header=None, nrows=5000
)

df_mnist_data = df_mnist.iloc[:, :-1]
df_mnist_labels = df_mnist.iloc[:, -1]

tsne_embedding = TSNE(n_components=2).fit_transform(df_mnist_data.values)
df_tsne_embedding = pd.DataFrame(data=tsne_embedding)

metric = DistanceMatrixBasedMetric(
    df_tsne_embedding, df_labels
)

print(metric.calculate())
```

2. `KMeans extension of distance matrix based metric` <br>
Calculation of distances between different classes is optimized by an approximation that mean distances between points from different classes are same as distances between centroids from KMeans.

```python
from sklearn.manifold import TSNE
from metrics.distance_matrix_and_kmeans_based_metric import DistanceMatrixAndKMeansBasedMetric


df_mnist = pd.read_csv(
    './datasets_with_labels/mnist.csv', header=None, nrows=5000
)

df_mnist_data = df_mnist.iloc[:, :-1]
df_mnist_labels = df_mnist.iloc[:, -1]

tsne_embedding = TSNE(n_components=2).fit_transform(df_mnist_data.values)
df_tsne_embedding = pd.DataFrame(data=tsne_embedding)

metric = DistanceMatrixAndKMeansBasedMetric(
    df_tsne_embedding, df_labels
)

print(metric.calculate())
```

3. `Thrustworthiness based metric` <br>
```python
from sklearn.manifold import TSNE
from metrics.trustworthiness_based_metric import TrustworthinessBasedMetric

df_mnist = pd.read_csv(
    './datasets_with_labels/mnist.csv', header=None, nrows=5000
)

df_mnist_data = df_mnist.iloc[:, :-1]

tsne_embedding = TSNE(n_components=2).fit_transform(df_mnist_data.values)
df_tsne_embedding = pd.DataFrame(data=tsne_embedding)

metric = TrustworthinessBasedMetric(
    df_mnist_data, df_tsne_embedding
)

print(metric.calculate())
```
Result:

```python
{
    'euclidean': {
        '5':   0.990,
        '10':  0.982,
        '15':  0.976,
        '30':  0.964,
        '50':  0.953,
        '100': 0.933,
        '150': 0.918,
        '300': 0.873,
        '500': 0.822
    },
    'cosine': {
        '5':   0.989,
        '10':  0.982,
        '15':  0.978,
        '30':  0.968,
        '50':  0.960,
        '100': 0.945,
        '150': 0.933,
        '300': 0.899,
        '500': 0.854
    }
}
```
