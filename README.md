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
final boolean shuffle = false;
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

''' Confusion Matrix

           Predicted     Predicted
           True          False
        ┌─────────────┬─────────────┐
Actual  │    78.2 %   │    21.8 %   │
True    │     (43)    │     (12)    │
        ├─────────────┼─────────────┤
Actual  │    47.6 %   │    61.0 %   │
False   │     (39)    │     (61)    │
        └─────────────┴─────────────┘

· Accuracy: 67.1 %
· Precision: 52.4 %
· Recall: 78.2 %
· F1: 62.8 %
'''

```