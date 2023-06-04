package se.pedramfk.portfolio.ml.ann.models;

import java.util.ArrayList;
import java.util.List;
import se.pedramfk.portfolio.ml.ann.layers.DenseLayer;
import se.pedramfk.portfolio.ml.utils.MatrixData;
import se.pedramfk.portfolio.ml.ann.activations.*;
import se.pedramfk.portfolio.ml.ann.layers.Layer;
import se.pedramfk.portfolio.ml.ann.losses.CrossEntropyLoss;
import se.pedramfk.portfolio.ml.utils.Matrix;


public class Network {

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
    private final Matrix getBackwardProp(Matrix loss, double learningRate) {

        for (int i = layers.size() - 1; i >= 0 ; i--) {
            loss = layers.get(i).backwardPropagate(loss, learningRate);
        }

        return loss;

    }

    /**
     * Predict values from given inputs.
     * @param x     input data to be predicted
     * @return      predicted values
     * @see         #getForwardProp(Matrix)
     */
    public final Matrix predict(Matrix x) {
        return Matrix.transpose(getForwardProp(Matrix.transpose(x)));
    }

    public static final double getLearningRate(int epoch) {

        final double etaStep = .01;
        final double etaMin = 1e-6;
        final double etaMax = 4e-1;

        double cycle = Math.floor(1.0 + (float) epoch / (2.0 + etaStep));

        double e = Math.abs((float) epoch / etaStep - 2.0 * cycle + 1.0);

        return etaMin + (etaMax - etaMin) * Math.max(0, 1.0 - e);

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

    private final double fitBatch(Matrix x, Matrix y, int epochs, int batchSize, boolean shuffle, double learningRate) {

        CrossEntropyLoss crossEntropyLoss = new CrossEntropyLoss();
        
        double squaredErrorLoss = 0.0;

        for (int i = 0; i < Math.floorDiv(x.rows, batchSize); i++) {

            Matrix xi, yi;

            if (shuffle) {

                Integer[] indices = MatrixData.getRandomIndices(batchSize);

                xi = Matrix.transpose(Matrix.getIndices(x, 0, indices));
                yi = Matrix.transpose(Matrix.getIndices(y, 0, indices));

            } else {

                xi = Matrix.transpose(Matrix.slice(x, 0, i * batchSize, (i + 1) * batchSize));
                yi = Matrix.transpose(Matrix.slice(y, 0, i * batchSize, (i + 1) * batchSize));

            }
            
            Matrix yiPred = getForwardProp(xi);

            Matrix yiLoss = Matrix.subtract(yiPred, yi);

            Matrix lossGrad = crossEntropyLoss.getLossGradient(yiPred, yi);
            //Matrix lossGrad = Matrix.subtract(yiPred, yi);

            squaredErrorLoss += Matrix.sum(Matrix.dot(yiLoss, yiLoss));
            
            getBackwardProp(lossGrad, learningRate);
            
        }

        return squaredErrorLoss / batchSize;

    }

    private final void fitEpoch(Matrix x, Matrix y, int epochs, int batchSize, boolean shuffle, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            double mse = fitBatch(x, y, epochs, batchSize, shuffle, learningRate);
         
            System.out.println(String.format("epoch = %d\tmse = %f", currentEpoch, mse));

        }

    }

    public final void fit(Matrix x, Matrix y, int epochs, int batchSize, boolean shuffle, double learningRate) {
        fitEpoch(x, y, epochs, batchSize, shuffle, learningRate);
    }

    public final void fit(Matrix x, Matrix y, int epochs, int batchSize, double learningRate) {
        fit(x, y, epochs, batchSize, false, learningRate);
    }


    public final void fit(Matrix xTrain, Matrix yTrain, Matrix xVal, Matrix yVal, int epochs, double learningRate) {

        final CrossEntropyLoss crossEntropyLoss = new CrossEntropyLoss();

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            //Matrix yMSE = new Matrix(x.rows, y.cols).initializeWithValue(0);
            //Matrix yPred = new Matrix(yTrain.rows, yTrain.cols);
            
            for (int i = 0; i < xTrain.rows; i++) {

                Matrix xi = Matrix.transpose(new Matrix(new double[][] { xTrain.get(i) }));
                Matrix yi = Matrix.transpose(new Matrix(new double[][] { yTrain.get(i) }));
                
                Matrix yiPred = getForwardProp(xi);

                Matrix lossGrad = crossEntropyLoss.getLossGradient(yiPred, yi);
                
                //yPred.set(i, yiPred.get(0));

                getBackwardProp(lossGrad, learningRate);
                
            }

            Matrix yPred = Matrix.transpose(getForwardProp(Matrix.transpose(xVal)));
            double mse = getMse(yVal, yPred);
            double acc = getAccuracy(yVal, yPred);

            System.out.println(
                String.format("Epoch = %d\tMSE = %f\t\tAccuracy = %f\tLearning-Rate = %f", 
                    currentEpoch, mse, acc, learningRate));

        }


    }

    public static final void main(String[] args) {

        double[][] xArray = {
            new double[] {0, 0}, 
            new double[] {0, 1}, 
            new double[] {1, 0}, 
            new double[] {1, 1}
        };

        double[][] yArray = {
            new double[] {0}, 
            new double[] {1}, 
            new double[] {1}, 
            new double[] {0}
        };

        Matrix x = new Matrix(xArray);
        Matrix y = new Matrix(yArray);

        Network network = new Network();
        network.addLayer(new DenseLayer(2, 3, new TanhActivation()));
        network.addLayer(new DenseLayer(3, 1, new TanhActivation()));

        //network.predict(x).print("Prediction");

        network.fit(x, y, x, y, 200000, 1e-4);
        //network.fitEpoch(x, y, 20000, 1, 1e-1);

        network.predict(x).print("Prediction");

    }
    
}
