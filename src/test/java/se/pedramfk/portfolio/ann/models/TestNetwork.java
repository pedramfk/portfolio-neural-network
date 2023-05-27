package se.pedramfk.portfolio.ann.models;

import se.pedramfk.portfolio.ann.activations.Activation;
import se.pedramfk.portfolio.ann.layers.dense.DenseLayer;
import se.pedramfk.portfolio.ann.utils.LoadData;
import se.pedramfk.portfolio.ann.utils.LossFunctions;
import se.pedramfk.portfolio.ann.utils.MatrixData;
import se.pedramfk.portfolio.ann.utils.LoadData.TrainAndTestData;


public final class TestNetwork {


    public static final void main(String[] args) throws Exception {

        final String path = "/Users/pedramfk/workspace/git/portfolio/neural-network/src/test/resources/pima-indians-diabetes.csv";

        final TrainAndTestData trainAndTestData = LoadData.loadTrainAndTestData(path, ",", .8);

        final double[][] xTrain = trainAndTestData.getTrainData().getNormalizedX();
        final double[][] yTrain = trainAndTestData.getTrainData().getY();

        final double[][] xTest = trainAndTestData.getTestData().getNormalizedX();
        final double[][] yTest = trainAndTestData.getTestData().getY();

        final Network network = new Network() {{
            addLayer(new DenseLayer(8, 6, Activation.TANH, Activation.DELTA_TANH));
            addLayer(new DenseLayer(6, 3, Activation.TANH, Activation.DELTA_TANH));
            addLayer(new DenseLayer(3, 1, Activation.SIGMOID, Activation.DELTA_SIGMOID));
            
        }};

        //network.predict(x).print("Prediction");

        network.fit(xTrain, yTrain, 2000, 1e-3);

        MatrixData yPred = network.predict(xTest);
        MatrixData yTrue = MatrixData.create(yTest);

        System.out.println(LossFunctions.getAccuracy(yTrue, yPred));

        //print("Prediction");
        //print("Actual");
        
    }
    
}
