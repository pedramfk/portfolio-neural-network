package se.pedramfk.portfolio.ann.layers.dense;

import se.pedramfk.portfolio.ann.activations.Activation;
import se.pedramfk.portfolio.ann.layers.Layer;
import se.pedramfk.portfolio.ann.utils.MatrixData;
import static se.pedramfk.portfolio.ann.utils.MatrixData.multiply;
import static se.pedramfk.portfolio.ann.utils.MatrixData.transpose;
import static se.pedramfk.portfolio.ann.utils.MatrixData.subtract;
import static se.pedramfk.portfolio.ann.utils.MatrixData.dot;
import java.util.function.Function;

public final class DenseLayer implements Layer {

    public final int n, m;
    
    private final MatrixData w, b;
    
    private MatrixData x;  // x * W
    private MatrixData a;  // f(x * W)

    private static Function<Double, Double> activation, deltaActivation;

    public DenseLayer(
        int inputSize, 
        int outputSize, 
        Function<Double, Double> activation, 
        Function<Double, Double> deltaActivation) {

        this.n = inputSize;
        this.m = outputSize;

        this.w = new MatrixData(inputSize, outputSize).initializeWithRandomValue();
        this.b = new MatrixData(1, outputSize).initializeWithValue(outputSize);

        DenseLayer.activation = activation;
        DenseLayer.deltaActivation = deltaActivation;

    }

    public MatrixData forwardPropagate(MatrixData input) {

        this.x = input;
        this.a = multiply(input, w).apply(DenseLayer.activation);

        return a;

        // output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        // output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))

    }

    /**
     * Backward Propagation.
     * 
     * @param error     MatrixData.subtract(yPred, yTrue)
     * @param lr        learning rate
     * @return propagated errors
     */
    public MatrixData backwardPropagate(MatrixData error, double lr) {

        final MatrixData delta = dot(a.copy().apply(deltaActivation), error); //error.copy().multiply(a.copy().transpose().apply(deltaActivation)).transpose();
        
        final MatrixData propagatedError = multiply(w, transpose(delta)).transpose();

        w.add(multiply(transpose(x), delta));

        return propagatedError;

    }

    public MatrixData backwardPropagate(MatrixData target, MatrixData output, double lr) {
        
        MatrixData err = subtract(target, output);

        MatrixData grad = multiply(output.copy().transpose().apply(deltaActivation), err).multiply(lr);

        grad.print();
        MatrixData deltaW = multiply(grad, w);
        deltaW.print();
        w.add(deltaW);
        b.add(grad);

        return multiply(err, transpose(w));

    }

    public MatrixData getWeights() {
        return this.w;
    }

    public MatrixData getBias() {
        return this.b;
    }
    
    public static final void main(String[] args) {

        MatrixData x = MatrixData.create(new double[][] {{3, 2, 1}});
        MatrixData y = MatrixData.create(new double[][] {{1, 3}});
        x.print("Matrix: x");
        y.print("Matrix: y");

        DenseLayer layer = new DenseLayer(3, 2, Activation.SIGMOID, Activation.DELTA_SIGMOID);
        
        layer.getWeights().print("Matrix: W");
        transpose(layer.getWeights()).print("Matrix: W^T");
        layer.getBias().print("Matrix: b");

        MatrixData yPred = layer.forwardPropagate(x);
        yPred.print("Matrix: yPred = X * W + b");

        //MatrixData yLoss = LossFunctions.getDeltaMSE(y, yPred);
        //yLoss.print("Matrix: yLoss = 2 * (yPred - yTrue) / N");

        //MatrixData yDelta = MatrixData.subtract(outpuData, new MatrixData(outpuData.getRows(), outpuData.getCols(), 1));
        MatrixData yDiff = MatrixData.create(new double[][] {{.4, .3}});

        //yDiff = layer.backwardPropagate(y, yPred, .1);
        layer.backwardPropagate(yDiff, .1).print("Matrix: de / dy");

        //layer.backwardPropagate(subtract(y, yPred), .1).print("Matrix: de / dy");


    }

    
}
