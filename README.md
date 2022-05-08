```bash
git clone https://gitlab.com/bminch/viskit.git
docker build -t viskit -f Dockerfile .
dockr run -it viskit /bin/bash
```

```bash
cd /opt/viskit/viskit_offline
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt 2500 2 1 1 0 0 0 "force-directed"
./viskit_offline "./datasets/mnist_data.csv" "./labels/mnist_labels.csv" "./graphs/mnist.bin" ./visualization.txt
```