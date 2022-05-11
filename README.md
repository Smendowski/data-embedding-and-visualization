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

## Examples
```bash
cd /opt/viskit/viskit_offline
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt
```
