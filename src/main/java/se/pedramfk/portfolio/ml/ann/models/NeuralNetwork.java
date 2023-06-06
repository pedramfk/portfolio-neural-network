package se.pedramfk.portfolio.ml.ann.models;

import java.util.ArrayList;
import java.util.List;

import se.pedramfk.portfolio.ml.ann.layers.Layer;
import se.pedramfk.portfolio.ml.ann.losses.LossFunction;
import se.pedramfk.portfolio.ml.ann.optimizers.Optimizer;
import se.pedramfk.portfolio.ml.ann.results.ModelResults;
import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.MatrixData;

public class NeuralNetwork {

    private static final String DEFAULT_LABEL = "Neural Network";

    private final LossFunction loss;
    private final Optimizer optimizer;

    private List<Layer> layers;

    String label;

    public NeuralNetwork(LossFunction loss, Optimizer optimizer, String label) {

        this.loss = loss;
        this.optimizer = optimizer;
        this.layers = new ArrayList<>();
        this.label = label;

    }

    public NeuralNetwork(LossFunction loss, Optimizer optimizer) {
        this(loss, optimizer, DEFAULT_LABEL);
    }

    public void addLayer(Layer layer){
        this.layers.add(layer);
    }

    public boolean validate() {
        return false;
    }

    public boolean compile() {
        return false;
    }

    public ModelResults fit(Matrix x, Matrix y, Matrix xVal, Matrix yVal, int epochs, int batchSize, boolean shuffle) {

        ModelResults results = new ModelResults(epochs);

        Integer[] indices = MatrixData.getRandomIndices(batchSize);

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            if (shuffle) {
                indices = MatrixData.getRandomIndices(batchSize);
            }
            
            Matrix xBatch = Matrix.getIndices(x, 0, indices);
            Matrix yBatch = Matrix.getIndices(y, 0, indices);

            for (int i = 0; i < batchSize; i++) {

                Matrix yiBatch = new Matrix(new double[][] { yBatch.get(i) });
                Matrix yiBatchPred = new Matrix(new double[][] { xBatch.get(i) });

                for (Layer layer: layers) {
                    yiBatchPred = layer.forwardPropagate(yiBatchPred);
                }
                
                Matrix gradient = loss.getLossGradient(yiBatchPred, yiBatch);
                for (int l = layers.size() - 1; l >= 0 ; l--) {
                    //gradient = this.optimizer.step(i, this.layers.get(l), gradient);
                }
                
            }

        }

        return results;

    }

    @Override
    public String toString() {
        return "";
    }

    public void print() {

    }


    
}

/*
Model: Pima Classifier

Layer       Type         Activation
     ┌────────────────┬──────────────┐
   1 ┼   Dense(8, 4)  │     ReLu     │
     ├────────────────┼──────────────┤
   2 ┼   Dense(4, 3)  │     Tanh     │
     ├────────────────┼──────────────┤
   3 ┼   Dense(2, 1)  │   Sigmoid    │  <--- Invalid Shape
     └────────────────┴──────────────┘

· Input Shape: [4 × N]
· Output Shape: [1 × N]
 */