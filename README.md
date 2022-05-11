```bash
git clone https://gitlab.com/bminch/viskit.git
docker build -t viskit -f Dockerfile .
docker run -it viskit /bin/bash
```

# Using viskit
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
