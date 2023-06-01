# Neural Network
A pure Java implementation of a neural network with no requirements of external libraries.

---

## Usage

```java
final String trainPath = "...";
final String testPath = "...";

final TrainAndTestData data = loadData(trainPath, testPath);
        
Matrix xTrain = new Matrix(data.getTrainData().getNormalizedX());
Matrix yTrain = new Matrix(data.getTrainData().getY());
        
Matrix xTest = new Matrix(data.getTestData().getNormalizedX());
Matrix yTest = new Matrix(data.getTestData().getY());

Network network = new Network() {{
    addLayer(new DenseLayer(8, 4, new TanhActivation()));
    addLayer(new DenseLayer(4, 2, new TanhActivation()));
    addLayer(new DenseLayer(2, 1, new SigmoidActivation()));
}};

final int epochs = 2000;
final int batchSize = 8;
final boolean shuffle = true;
final double learningRate = 1e-3;

Results results = network.fit(
    xTrain, 
    yTrain, 
    epochs, 
    batchSize, 
    shuffle, 
    learningRate);

Matrix yPred = network.predict(xTest);

BinaryClassificationResults res = new BinaryClassificationResults(yTest, yPred);
        
res.getConfusionMatrix().print();
```

```text
~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~

            Predicted     Predicted
            True          False
         ┌─────────────┬─────────────┐
 Actual  │    65.5 %   │    34.5 %   │
 True    │     (36)    │     (19)    │
         ├─────────────┼─────────────┤
 Actual  │    30.8 %   │    84.0 %   │
 False   │     (16)    │     (84)    │
         └─────────────┴─────────────┘

 · Accuracy: 77.4 %
 · Precision: 69.2 %
 · Recall: 65.5 %
 · F1: 67.3 %

 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

---

## Modeling

### Dimensions

- *Input data $X$ of dimension $(d_0 \times n_b)$.*
- *Target data $Y$ of dimension $(d_K \times n_b)$.*
- *Layer weight $W_k$ of dimension $(d_k \times d_{k-1})$.*
- *Layer bias $b_k$ of dimension $(d_k \times n_b)$.*

### Forward Propagation

- *Layer indices $k = 1, 2, \dots, K$.*
- *Layer input $X_k = A_{k-1}$.*
- *Linear layer output $Z_k = W_k X_k + b_k$.*
- *Non-linear layer output $A_k = f\left(Z_k\right)$.*
  - *E.g. sigmoid activation: $f(z) = \frac{1}{1 + e^{-z}}$*
- *Predictions $\hat{Y} = A_K$.*

### Cross-Entropy Loss

- *$n_b = 1 \rightarrow L_{\mathcal{c}} = - \ln (Y \cdot A_K).$*
- *$n_b > 1 \rightarrow L_{\mathcal{c}} =- \ln \frac{1}{n_b} \sum Y^TA_{K}$.*

### Quadratic Loss

- *$n_b = 1 \rightarrow L_{\mathcal{q}} = \frac{1}{2} (Y - A_K)(Y - A_K)^T$*
- *$n_b > 1 \rightarrow L_{\mathcal{q}} = \frac{1}{2n_b} (Y - A_K)(Y - A_K)^T$*
- $\nabla_A L_q(\cdot) = - (Y - A_K)$

### Backward Propagation

- *Output gradient loss $\nabla_A L_q (\cdot) \vert_{k=K} = - (Y - A_K)$.*
- *Layer gradient loss $\delta_k = W_{k+1}^T \delta_{k+1} \cdot \nabla_Z f (\cdot) \vert_{k=k}$.*
  - $\rightarrow \nabla_W L (\cdot) \vert_{k=k} = A_{k-1} \delta_k = X_k\delta_k$
  - $\rightarrow \nabla_b L (\cdot) \vert_{k=k} = \delta_k$
  - $\rightarrow W_k^{(i+1)} = W_k^{(i)} - \eta X_k\delta_k$
  - $\rightarrow b_k^{(i+1)} = b_k^{(i)} - \eta X_k\delta_k$

---

## TODO

[x] Cross-Entropy loss

[ ] Regularization

[ ] Batch Normalization

[ ] Weight & Bias Initialization methods

[ ] Dropout Layer

[ ] Adaptive Learning Rate

[ ] Wrapper for Train Results

[ ] Unit Tests

[ ] Plots