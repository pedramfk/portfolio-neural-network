package se.pedramfk.portfolio.ml.ann;

import se.pedramfk.portfolio.ml.utils.MatrixData;
import se.pedramfk.portfolio.ml.utils.MatrixData.InputAndOutputData;
import se.pedramfk.portfolio.ml.utils.MatrixData.TrainAndTestData;
import se.pedramfk.portfolio.ml.ann.activations.SigmoidActivation;
import se.pedramfk.portfolio.ml.ann.activations.TanhActivation;
import se.pedramfk.portfolio.ml.ann.layers.DenseLayer;
import se.pedramfk.portfolio.ml.ann.models.Network;
import se.pedramfk.portfolio.ml.metrics.BinaryClassificationResults;
import se.pedramfk.portfolio.ml.utils.Matrix;

public class TestNetwork {

    public static final double getAccuracy(Matrix yTrue, Matrix yPred) {

        int n = yTrue.rows;
        double correct = 0;

        for (int i = 0; i < n; i++) {
            double pred = yPred.get(i, 0) > .5 ? 1.0 : 0.0;
            double actual = yTrue.get(i, 0) > .5 ? 1.0 : 0.0;
            if (pred == actual) correct++;
        }

        return correct / n;

    }

    public static final double getTPr(Matrix yTrue, Matrix yPred) {

        int n = yTrue.rows;

        double nTp = 0.0;
        double nFn = 0.0;

        for (int i = 0; i < n; i++) {
            double pred = yPred.get(i, 0) > .5 ? 1.0 : 0.0;
            double actual = yTrue.get(i, 0) > .5 ? 1.0 : 0.0;
            if ((pred == 1.) && (actual == 1.)) nTp++;
            else if ((pred == 0.) && (actual == 1.)) nFn++;
        }

        return nTp / (nTp + nFn);

    }

    public static final double getFNr(Matrix yTrue, Matrix yPred) {

        int n = yTrue.rows;

        double nTp = 0.0;
        double nFn = 0.0;

        for (int i = 0; i < n; i++) {
            double pred = yPred.get(i, 0) > .5 ? 1.0 : 0.0;
            double actual = yTrue.get(i, 0) > .5 ? 1.0 : 0.0;
            if ((pred == 1.) && (actual == 1.)) nTp++;
            else if ((pred == 0.) && (actual == 1.)) nFn++;
        }

        return nTp / (nTp + nFn);

    }

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

        final double[][] xTrainArray = trainAndTestData.getTrainData().getNormalizedX();
        final double[][] yTrainArray = trainAndTestData.getTrainData().getY();

        final double[][] xTestArray = trainAndTestData.getTestData().getNormalizedX();
        final double[][] yTestArray = trainAndTestData.getTestData().getY();

        Matrix xTrain = new Matrix(xTrainArray);
        Matrix xTest = new Matrix(xTestArray);
        Matrix yTrain = new Matrix(yTrainArray);
        Matrix yTest = new Matrix(yTestArray);

        final Network network = new Network() {{
            addLayer(new DenseLayer(8, 4, new TanhActivation()));
            addLayer(new DenseLayer(4, 2, new TanhActivation()));
            addLayer(new DenseLayer(2, 1, new SigmoidActivation()));
        }};

        //network.fit(xTrain, yTrain, 2000, 1e-2);
        network.fitEpoch(xTrain, yTrain, 8000, 8, 1e-4);

        Matrix yPred = network.predict(xTest, yTest);

        BinaryClassificationResults res = new BinaryClassificationResults(yTest, yPred);
        
        res.getConfusionMatrix().print();
        
    }
    
}
