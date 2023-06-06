package se.pedramfk.portfolio.ml.ann;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.MatrixData;
import se.pedramfk.portfolio.ml.utils.MatrixData.InputAndOutputData;
import se.pedramfk.portfolio.ml.utils.MatrixData.TrainAndTestData;
import se.pedramfk.portfolio.ml.ann.activations.*;
import se.pedramfk.portfolio.ml.ann.models.Network;
import se.pedramfk.portfolio.ml.ann.layers.DenseLayer;
import se.pedramfk.portfolio.ml.ann.losses.BinaryCrossEntropyLoss;
import se.pedramfk.portfolio.ml.metrics.BinaryClassificationResults;


public class TestNetwork {

    public static final InputAndOutputData loadData(String path) throws Exception {
        return MatrixData.loadInputAndOutputData(path, ",");
    }

    public static final TrainAndTestData loadData(String trainPath, String testPath) throws Exception {

        InputAndOutputData trainData = MatrixData.loadInputAndOutputData(trainPath, ",");
        InputAndOutputData testData = MatrixData.loadInputAndOutputData(testPath, ",");

        double[] trainMaxVals = MatrixData.getMaxValues(trainData.getX());
        double[] testMaxVals = MatrixData.getMaxValues(testData.getX());
        double[] maxVals = new double[trainMaxVals.length];

        for (int i = 0; i < trainMaxVals.length; i++) {
            maxVals[i] = Math.max(trainMaxVals[i], testMaxVals[i]);
        }

        trainData.setMaxValues(maxVals);
        testData.setMaxValues(maxVals);

        return new TrainAndTestData(trainData, testData);

    }

    public static final void main(String[] args) throws Exception {

        final String trainPath = "/Users/pedramfk/workspace/git/portfolio/neural-network/src/test/resources/pima-indians-diabetes/train.csv";
        final String testPath = "/Users/pedramfk/workspace/git/portfolio/neural-network/src/test/resources/pima-indians-diabetes/test.csv";

        final TrainAndTestData trainAndTestData = loadData(trainPath, testPath);
        
        final Matrix xTrain = new Matrix(trainAndTestData.getTrainData().getNormalizedX());
        final Matrix yTrain = new Matrix(trainAndTestData.getTrainData().getY());
        
        final Matrix xTest = new Matrix(trainAndTestData.getTestData().getNormalizedX());
        final Matrix yTest = new Matrix(trainAndTestData.getTestData().getY());

        final Network network = new Network(new BinaryCrossEntropyLoss()) {{
            addLayer(new DenseLayer(8, 7, new SigmoidActivation()));
            addLayer(new DenseLayer(7, 3, new SigmoidActivation()));
            addLayer(new DenseLayer(3, 1, new SigmoidActivation()));
        }};
        
        network.fit(xTrain, yTrain, xTest, yTest, 3000, 4e-2);
        //network.fit(xTrain, yTrain, xTest, yTest, 10000, 8, false, 1e-4);

        Matrix yPred = network.predict(xTest);

        BinaryClassificationResults res = new BinaryClassificationResults(yTest, yPred);
        
        res.getConfusionMatrix().print();
        
    }
    
}
