package se.pedramfk.portfolio.ml.ann.models;

import java.util.List;
import java.util.ArrayList;
import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.ann.layers.Layer;
import se.pedramfk.portfolio.ml.ann.losses.LossFunction;


public class Network {

    private final LossFunction lossFunction;

    public Network(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    private final List<Layer> layers = new ArrayList<>();

    /**
     * Add a layer to this network.
     * @param layer     a layer implementation
     */
    public final void addLayer(Layer layer) {

        this.layers.add(layer);

    }

    /**
     * Forward propagate over all {@link #layers}.
     * @param x     input data to be forwarded through all layers
     * @return      output data from last layer
     * @see         #addLayer(Layer)
     */
    private final Matrix getForwardProp(Matrix x) {

        for (Layer layer: layers) {
            x = new Matrix(layer.forwardPropagate(x));
        }

        return x;

    }

    /**
     * Backward propagate over all {@link #layers}.
     * @param loss              loss to be propagated through all layers in reverse order
     * @param learningRate      learning-rate used to tune layer parameters
     * @return                  final propagated loss gradients from first layer
     */
    private final Matrix getBackwardProp(Matrix lossGrad, double learningRate) {

        for (int i = layers.size() - 1; i >= 0 ; i--) {
            lossGrad = layers.get(i).backwardPropagate(lossGrad, learningRate);
        }

        return lossGrad;

    }

    /**
     * Predict values from given inputs.
     * @param x     input data to be predicted
     * @return      predicted values
     * @see         #getForwardProp(Matrix)
     */
    public final Matrix predict(Matrix x) {

        return getForwardProp(x);

    }

    public static final double getMse(Matrix yTrue, Matrix yPred) {

        double sum = 0.0;

        for (int i = 0; i < yTrue.rows; i++) {
            sum += Math.pow(yPred.get(i, 0) - yTrue.get(i, 0), 2);
        }

        return sum / yTrue.rows;

    }

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

    public final void fit(Matrix xTrain, Matrix yTrain, Matrix xVal, Matrix yVal, int epochs, int batchSize, boolean shuffle, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            for (int i = 0; i < Math.floorDiv(xTrain.rows, batchSize); i++) {

                Matrix xi = Matrix.slice(xTrain, 0, i * batchSize, (i + 1) * batchSize);
                Matrix yi = Matrix.slice(yTrain, 0, i * batchSize, (i + 1) * batchSize);
                
                Matrix yiPred = predict(xi);

                Matrix lossGrad = lossFunction.getLossGradient(yi, yiPred);
                
                getBackwardProp(lossGrad, learningRate);
                
            }

            Matrix yTrainPred = predict(xTrain);
            Matrix yValPred = predict(xVal);

            double trainMse = getMse(yTrain, yTrainPred);
            double valMse = getMse(yVal, yValPred);

            double trainAcc = getAccuracy(yTrain, yTrainPred);
            double valAcc = getAccuracy(yVal, yValPred);

            double trainLoss = lossFunction.getLoss(yTrain, yTrainPred);
            double valLoss = lossFunction.getLoss(yVal, yValPred);

            System.out.println(String.format(
                "Epoch = %d | Train: Loss = %.4f, MSE = %.4f Accuracy = %.4f | Validation: Loss = %.4f, MSE = %.4f, Accuracy = %.4f", 
                currentEpoch, trainLoss, trainMse, trainAcc, valLoss, valMse, valAcc));

        }

    }

    public final void fit(Matrix xTrain, Matrix yTrain, Matrix xVal, Matrix yVal, int epochs, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            double lr = learningRate * Math.exp(-.0001 * currentEpoch);
            
            for (int i = 0; i < xTrain.rows; i++) {

                Matrix xi = new Matrix(new double[][] { xTrain.get(i) });
                Matrix yi = new Matrix(new double[][] { yTrain.get(i) });
                
                Matrix yiPred = getForwardProp(xi);

                Matrix lossGrad = lossFunction.getLossGradient(yi, yiPred);

                getBackwardProp(lossGrad, lr);
                
            }

            Matrix yTrainPred = getForwardProp(xTrain);
            Matrix yValPred = getForwardProp(xVal);

            double trainMse = getMse(yTrain, yTrainPred);
            double valMse = getMse(yVal, yValPred);

            double trainAcc = getAccuracy(yTrain, yTrainPred);
            double valAcc = getAccuracy(yVal, yValPred);

            double trainLoss = lossFunction.getLoss(yTrain, yTrainPred);
            double valLoss = lossFunction.getLoss(yVal, yValPred);

            System.out.println(String.format(
                "Epoch=%d  |  Train  ->  Loss=%.4f, MSE=%.4f Acc=%.4f  |  Val  ->  Loss=%.4f, MSE=%.4f, Acc=%.4f  |  eta=%.5f", 
                currentEpoch, trainLoss, trainMse, trainAcc, valLoss, valMse, valAcc, lr));

            if (valAcc >= 0.800) break;

        }


    }
    
}
