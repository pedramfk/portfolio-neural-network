package se.pedramfk.portfolio.ml.ann.models;

import java.util.ArrayList;
import java.util.List;
import se.pedramfk.portfolio.ml.ann.layers.DenseLayer;
import se.pedramfk.portfolio.ml.utils.MatrixData;
import se.pedramfk.portfolio.ml.ann.activations.*;
import se.pedramfk.portfolio.ml.ann.layers.Layer;
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
        return getForwardProp(x);
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

    public static final double getMse(Matrix yTrue, Matrix yPred) {

        double sum = 0.0;

        for (int i = 0; i < yTrue.rows; i++) {
            sum += Math.pow(yPred.get(i, 0) - yTrue.get(i, 0), 2);
        }

        return sum / yTrue.rows;

    }

    private final double fitBatch(Matrix x, Matrix y, int epochs, int batchSize, boolean shuffle, double learningRate) {

        double squaredErrorLoss = 0.0;

        for (int i = 0; i < Math.floorDiv(x.rows, batchSize); i++) {

            Matrix xi, yi;

            if (shuffle) {

                Integer[] indices = MatrixData.getRandomIndices(batchSize);

                xi = Matrix.getIndices(x, 0, indices);
                yi = Matrix.getIndices(y, 0, indices);

            } else {

                xi = Matrix.slice(x, 0, i * batchSize, (i + 1) * batchSize);
                yi = Matrix.slice(y, 0, i * batchSize, (i + 1) * batchSize);

            }

            
            Matrix yiPred = getForwardProp(xi);

            Matrix error = Matrix.subtract(yi, yiPred);

            squaredErrorLoss += Matrix.sum(Matrix.multiply(error, Matrix.transpose(error)));
            
            Matrix yiLoss = error.multiply(-2);

            getBackwardProp(yiLoss, learningRate);
            
        }

        return squaredErrorLoss;

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

    /*
    public final void fit(Matrix x, Matrix y, int epochs, int batchSize, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            Matrix yPred = new Matrix(y.rows, y.cols);
            
            for (int i = 0; i < x.rows; i++) {

                Matrix xi = new Matrix(new double[][] { x.get(i) });
                Matrix yi = new Matrix(new double[][] { y.get(i) });
                
                Matrix yiPred = getForwardProp(xi);
                
                Matrix yiLoss = Matrix.subtract(yi, yiPred).multiply(-2);
                //MatrixData.subtract(yTrue, yPred).multiply(-2)
                yPred.set(i, yiPred.get(0));

                getBackwardProp(yiLoss, learningRate);
                                
                //MatrixData yPred = getForwardProp(new MatrixData(x[i]).transpose());
                //MatrixData yTarget = new MatrixData(y[i]).transpose();

                //double mse = LossFunctions.getMSE(yTarget, yPred);
                //yMSE.set(i, 0, mse);
                
                //yPred[i] = output.get(0);
                //mse += LossFunctions.getMSE(y[i], yPred.get(i));
                
                // Backward Propagate
                
                //getBackwardProp(LossFunctions.getDeltaMSE(yActual, yPred), lr);
                
            }

            double mse = getMse(y, yPred);
            //System.out.println(String.format("epoch = %d\terror = %f", currentEpoch, yMSE.get(0)[x.rows - 1]));
            System.out.println(String.format("epoch = %d\terror = %f", currentEpoch, mse));

        }


    } */

    public final void fit(Matrix x, Matrix y, int epochs, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            //Matrix yMSE = new Matrix(x.rows, y.cols).initializeWithValue(0);
            Matrix yPred = new Matrix(y.rows, y.cols);
            
            for (int i = 0; i < x.rows; i++) {

                Matrix xi = new Matrix(new double[][] { x.get(i) });
                Matrix yi = new Matrix(new double[][] { y.get(i) });
                
                Matrix yiPred = getForwardProp(xi);
                
                Matrix yiLoss = Matrix.subtract(yi, yiPred).multiply(-2);
                //MatrixData.subtract(yTrue, yPred).multiply(-2)
                yPred.set(i, yiPred.get(0));

                getBackwardProp(yiLoss, learningRate);
                                
                //MatrixData yPred = getForwardProp(new MatrixData(x[i]).transpose());
                //MatrixData yTarget = new MatrixData(y[i]).transpose();

                //double mse = LossFunctions.getMSE(yTarget, yPred);
                //yMSE.set(i, 0, mse);
                
                //yPred[i] = output.get(0);
                //mse += LossFunctions.getMSE(y[i], yPred.get(i));
                
                // Backward Propagate
                
                //getBackwardProp(LossFunctions.getDeltaMSE(yActual, yPred), lr);
                
            }

            double mse = getMse(y, yPred);
            //System.out.println(String.format("epoch = %d\terror = %f", currentEpoch, yMSE.get(0)[x.rows - 1]));
            System.out.println(String.format("epoch = %d\terror = %f", currentEpoch, mse));

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

        network.fit(x, y, 200000, 1e-4);
        //network.fitEpoch(x, y, 20000, 1, 1e-1);

        network.predict(x).print("Prediction");

    }
    
}
