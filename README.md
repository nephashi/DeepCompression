An implementation of [Iterative Pruning](https://arxiv.org/abs/1506.02626), current on mnist only.

Thanks [this repository](https://github.com/garion9013/impl-pruning-TF)

## Usage
### Iterative Pruning
```
cd mnist_iterative_pruning
python iterative_prune.py -1 -2 -3
```
this would train a convolution model on mnist. Then do pruning on fc layer and retraining for 20 times. Finally fc layers would be transformed to a sparse format and saved.

## Performance

we have a pretty good pruning performance, keeping accuracy at 0.987 while pruning 99.77% weights in fc layer.

|weight kept ratio|accuracy|
|-----------------|--------|
|1                |0.99    |
|0.7              |0.991   |
|0.49             |0.993   |
|0.24             |0.994   |
|0.117            |0.993   |
|0.057            |0.994   |
|0.013            |0.993   |
|0.009            |0.992   |
|0.0047           |0.99    |
|0.0023           |0.987   |
|0.0016           |0.889   |
|0.0011           |0.886   |
|0.00079          |0.677   |
|0.00056          |0.409   |

in term of inference time, dense vs sparse: 1.47 vs 0.68
