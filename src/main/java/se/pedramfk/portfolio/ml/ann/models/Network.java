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

    public final void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    private final Matrix getForwardProp(Matrix x) {
        for (Layer layer: layers) {
            x = new Matrix(layer.forwardPropagate(x));
        }
        return x;
    }

    private final Matrix getBackwardProp(Matrix loss, double learningRate) {
        for (int i = layers.size() - 1; i >= 0 ; i--) {
            Layer currentLayer = layers.get(i);
            loss = currentLayer.backwardPropagate(loss, learningRate);
        }
        return loss;
    }

    public final Matrix predict(Matrix x, Matrix y) {
        
        Matrix predictions = new Matrix(x.rows, y.cols);

        for (int i = 0; i < x.rows; i++) {

            Matrix xi = new Matrix(new double[][] { x.get(i) });
            Matrix yiPred = getForwardProp(xi);

            predictions.set(i, yiPred.get(0));

        }

        return predictions;

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

    public final void fitEpoch(Matrix x, Matrix y, int epochs, int batchSize, boolean shuffle, double learningRate) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            double mse = fitBatch(x, y, epochs, batchSize, shuffle, learningRate);
         
            System.out.println(String.format("epoch = %d\tmse = %f", currentEpoch, mse));

        }

    }

    public final void fitEpoch(Matrix x, Matrix y, int epochs, int batchSize, double learningRate) {
        fitEpoch(x, y, epochs, batchSize, false, learningRate);
    }

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


    }

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

        //network.fit(x, y, 1000, 1e-1);
        network.fitEpoch(x, y, 1000, 2, 1e-1);

        network.predict(x, y).print("Prediction");

    }
    
}
