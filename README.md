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
Actual  │    61.8 %   │    38.2 %   │
True    │     (34)    │     (21)    │
        ├─────────────┼─────────────┤
Actual  │    20.9 %   │    91.0 %   │
False   │      (9)    │     (91)    │
        └─────────────┴─────────────┘

· Accuracy: 80.6 %
· Precision: 79.1 %
· Recall: 61.8 %
· F1: 69.4 %

 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

---

## Modeling

### Dimensions

- **Number of layer:** $K$
- **Batch sample size:** $N$
- **Input dimension:** $[M \times N]$
- **Output dimension:** $[D \times N]$
- **Layer weight dimension:** $[d_{k} \times d_{k - 1}]$
- **Layer bias dimension:** $[d_{k} \times N]$
- **Input data:** $x$
- **Output data:** $y$
- **Layer weight:** $W_k$
- **Layer bias:** $b_k$
- **Layer activation function:** $f_k(\cdot)$
- **Layer input:** $a_{k-1}$
- **Layer output:** $a_K$

### Layers - Forward Propagation

- **Linear output:** $z_k = W_k a_{k-1} + b_k$
- **Non-linear output:** $a_k = f\left(z_k\right)$
  - *E.g. sigmoid activation:* $f(z) = \frac{1}{1 + e^{-z}}$
- **Predictions:** $\hat{y} = a_K = f_K(f_{K-1}(\dots f_1(W_1x + b_1)))$
- **Regularization:** $R(\cdot)$
  - *E.g. quadratic weight cost function:* $R_k(\cdot) = \lambda \sum_{(i, j)} \vert\vert W_k^{(i, j)} \vert\vert^2$
- **Loss:** $L(\cdot)$
  - *E.g. cross-entropy loss:* $L_C(\cdot) = - \sum_{i = 1}^{N-1}y \cdot \ln a_K$
- **Cost Function:** $C(\cdot) = L(\cdot) + R(\cdot)$
  - *E.g.:* $C(\cdot) = - \sum_{i = 1}^{N-1}y \cdot \ln a_K + \lambda \sum_{k=1}^{K} \sum_{(i, j)} \vert\vert W_k^{(i, j)} \vert\vert^2$

$$
a_k = f \left(
\underbrace{
\begin{bmatrix}
        W_k^{(0, 0)} & \dots & W_k^{(0, d_{k-1})}\\
        \vdots & \ddots & \vdots\\
        W_k^{(d_{k}, 0)} & \dots & W_k^{(d_{k}, d_{k-1})}
\end{bmatrix}}_{Wk}
\overbrace{
\begin{bmatrix}
        x_k^{(0, 0)} & \dots & x_k^{(0, n_b)}\\
        \vdots & \ddots & \vdots\\
        x_k^{(d_{k-1}, 0)} & \dots & x_k^{(d_{k-1}, n_b)}
\end{bmatrix}}^{x_k = a_{k-1}} + 
\underbrace{
\begin{bmatrix}
        b_k^{(0, 0)} & \dots & b_k^{(0, n_b)}\\
        \vdots & \ddots & \vdots\\
        b_k^{(d_{k}, 0)} & \dots & x_k^{(d_{k}, n_b)}
\end{bmatrix}}_{b_k}
\right) \\
 = \begin{bmatrix}
        a_k^{(0, 0)} & \dots & a_k^{(0, n_b)}\\
        \vdots & \ddots & \vdots\\
        a_k^{(d_{k}, 0)} & \dots & a_k^{(d_{k}, n_b)}
\end{bmatrix}
$$
### Layers - Backward Propagation

- **Output loss gradient:** $\delta_K = \nabla_a L_q (y, a_K)$
- **Layer loss gradient:** $\delta_k = W_{k+1}^T \delta_{k+1} \cdot \nabla_z f (z_k) \vert_{\forall k < K}$
  - $\rightarrow \nabla_W L (\cdot) \vert_{k=k} = a_{k-1} \delta_k = x_k\delta_k$
  - $\rightarrow \nabla_b L (\cdot) \vert_{k=k} = \delta_k$
  - $\rightarrow W_k^{(i+1)} = W_k^{(i)} - \eta \times x_k\delta_k$
  - $\rightarrow b_k^{(i+1)} = b_k^{(i)} - \eta \times x_k\delta_k$

$C = L(y, f(z_k)) + R(W_k)$

$C = L(y, \hat{y}) + \lambda \sum_{i=1}^{K} \sum_{(i, j) \in W_k} \vert\vert W_k^{(i, j)} \vert\vert^2$

$\frac{\partial C}{\partial W_k} 
        = \frac{\partial C}{\partial a_K} \times 
          \frac{\partial a_K}{\partial z_K} \times 
          \frac{\partial z_K}{\partial W_K} \times 
          \frac{\partial z_{K-1}}{\partial W_{K-1}} \times \dots \times 
          \frac{\partial z_k}{\partial W_k}$

$L(y, \hat{y}) = - \frac{1}{n_b} \sum_{j=0}^{n_b - 1} y_{i, j=j} \ln \hat{y}_{i, j=j}$

$a_K = f(z_K) = 1 / (1 + e^{-z_K})$

$\nabla_{a_K} C = - \frac{1}{n_b} y / a_K$

$\nabla_{z_K} a_K = f(z_K) \times \left( 1 - f(z_K) \right) = a_K \left( 1 - a_K \right)$

$\nabla_{W_K} z_K = a_{K-1}$

$\nabla_{W_K} C = - y(1 - a_K)x_K + 2 \lambda \sum \vert\vert W_k \vert\vert$

---

## TODO

[X] Cross-Entropy loss
 - [X] Binary Cross-Entropy

[X] Regularization
 - [X] Weight Matrices Cost
 - [ ] Dropout Layer
 - [ ] Batch Normalization

[ ] Weight & Bias Initialization methods

[ ] Adaptive Learning Rate
 - [X] Exponential Decay

[ ] Unit Tests

[ ] Nice to have
 - [ ] Wrapper for Train Results
 - [ ] Plots