# CGL-AD
A Dynamic Provenance Graph-Based Detector for Advanced Persistent Threats

## Dependencies

+ Python 3.8.18
+ Pytorch 1.8.1
+ PyG 2.0.0
+ Scikit-learn 1.3.1

## **Dataset**

+ **Streamspot** dataset is available at [streamspot](https://github.com/sbustreamspot/sbustreamspot-data) , you need to download `all.tar.gz`.
+ **Camflow-apt** and **Shellshock** dataset are released by [Unicorn](https://arxiv.org/abs/2001.01525) ,  can be found at [unicorn-data](https://github.com/margoseltzer). You ned to download `camflow-benign-*` and `camflow-attack-*`.

## Run

### parse

**For frequency estimation（graph sketching）**

```
# for single graph
parser/streamspot/parse_fast.py
parser/camflow/parse.py
```

**For temporal graph learning**

```
parser/streamspot/parse_temporal.py
parser/camflow/parse_temporal.py
```

### graph representation

#### frequency estimation

Code in the analyzer folder is adapted from [crimson-unicorn/analyzer](https://github.com/crimson-unicorn/analyzer)

```
cd CGL-AD/analyzer/
make sw
python gen_analyse_sh.py   # generate scripts for batch processing
chmod 777 analyse_{data}.sh
./analyse_{data}.sh
```

#### temporal graph learning

```
cd CGL-AD/modeler/{dataset}
python pretrain.py   # pretrain TGN
python embedding.py   # TGN inference
```

### anomaly detection

```
cd CGL-AD/modeler/{dataset}
python rcnn.py    # Bi-RCNN and detection
```

> The modelr folder contains the trained models.