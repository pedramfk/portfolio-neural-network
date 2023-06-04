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
Actual  │    60.0 %   │    40.0 %   │
True    │     (33)    │     (22)    │
        ├─────────────┼─────────────┤
Actual  │    21.4 %   │    91.0 %   │
False   │      (9)    │     (91)    │
        └─────────────┴─────────────┘

· Accuracy: 80.0 %
· Precision: 78.6 %
· Recall: 60.0 %
· F1: 68.0 %

 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

---

## Modeling

### Dimensions

- **Input data:** $x \in \mathbb{R}^{[ d_0 \times n_b ]}$
- **Target data:** $y \in \mathbb{R}^{[ d_K \times n_b ]}$
- **Layer weight:** $W_k \in \mathbb{R}^{[ d_{k} \times d_{k-1} ]}$
- **Layer bias:** $b_k \in \mathbb{R}^{[ d_k \times n_b ]}$

### Forward Propagation

- **Layer indices:** $k = 1, 2, \dots, K$
- **Layer input:** $x_k = a_{k-1}$
- **Linear layer output:** $z_k = W_k x_k + b_k$
- **Non-linear layer output:** $a_k = f\left(z_k\right)$
  - *E.g. sigmoid activation:* $f(z) = \frac{1}{1 + e^{-z}}$
- **Predictions:** $\hat{y} = a_K$.

$$
a_k = f \left (
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

### Cross-Entropy Loss

- $n_b = 1 \rightarrow L_{\mathcal{c}} \left(y, \hat{y} \right) = - y \cdot \ln \hat{y}$
- $n_b > 1 \rightarrow L_{\mathcal{c}} \left(y, \hat{y} \right) = - \frac{1}{n_b} \sum_i y^T \ln \hat{y}$

### Quadratic Loss

- $n_b = 1 \rightarrow L_{\mathcal{q}} \left(y, \hat{y} \right) = \frac{1}{2} (y - \hat{y})(y - \hat{y})^T$
- $n_b > 1 \rightarrow L_{\mathcal{q}} \left(y, \hat{y} \right) = \frac{1}{2n_b} (y - \hat{y})(y - \hat{y})^T$

### Backward Propagation

- **Output loss gradient:** $\delta_K = \nabla_a L_q (y, a_K)$
- **Layer loss gradient:** $\delta_k = W_{k+1}^T \delta_{k+1} \cdot \nabla_z f (z_k) \vert_{\forall k < K}$
  - $\rightarrow \nabla_W L (\cdot) \vert_{k=k} = a_{k-1} \delta_k = x_k\delta_k$
  - $\rightarrow \nabla_b L (\cdot) \vert_{k=k} = \delta_k$
  - $\rightarrow W_k^{(i+1)} = W_k^{(i)} - \eta \times x_k\delta_k$
  - $\rightarrow b_k^{(i+1)} = b_k^{(i)} - \eta \times x_k\delta_k$

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