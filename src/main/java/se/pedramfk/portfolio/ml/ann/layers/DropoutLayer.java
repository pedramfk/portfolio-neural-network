package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;


public class DropoutLayer implements Layer {

    private final double rate;

    private Matrix mask;

    public DropoutLayer(double rate) {
        this.rate = rate;
    }

    @Override
    public Matrix forwardPropagate(Matrix input) {
        Matrix mask = new Matrix(input.rows, input.cols).initializeWithValue(1);
        this.mask = Matrix.replace(mask, 0, (int) Math.round(rate * mask.rows * mask.cols));
        return Matrix.multiply(input, this.mask);
    }

    @Override
    public Matrix backwardPropagate(Matrix error, double learningRate) {
        return Matrix.multiply(this.mask, error);
    }
    
}
