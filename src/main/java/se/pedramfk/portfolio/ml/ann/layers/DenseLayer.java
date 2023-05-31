package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;
import static se.pedramfk.portfolio.ml.utils.Matrix.*;

import se.pedramfk.portfolio.ml.ann.activations.Activation;
import se.pedramfk.portfolio.ml.ann.activations.SigmoidActivation;


public final class DenseLayer implements HiddenLayer {

    private final int m, n;

    private Matrix w, b;

    private Matrix x, a;

    private final Activation activationLayer;

    public DenseLayer(int n, int m, Activation activationLayer) {
        
        this.n = n;
        this.m = m;

        this.w = new Matrix(n, m).initializeWithRandomValue(.1);
        this.b = new Matrix(1, m).initializeWithRandomValue(.1);

        this.activationLayer = activationLayer;

    }


    @Override
    public Matrix forwardPropagate(Matrix input) {

        this.x = input.copy();
        //this.a = activationLayer.getActivation(add(multiply(this.x, this.w), repeat(this.b, 0, input.cols)));
        
        this.a = activationLayer.getActivation(add(multiply(this.x, this.w), repeat(this.b, 0, input.rows)));

        return this.a;

    }

    @Override
    public Matrix backwardPropagate(Matrix outputError, double learningRate) {

        Matrix dA = dot(outputError, activationLayer.getActivationGradient(this.a));
        
        Matrix propagatedError = multiply(dA, transpose(this.w));

        Matrix dW = multiply(transpose(this.x), dA);
        Matrix db = mean(dA, 0);

        this.w = subtract(this.w, dW.multiply(learningRate));
        this.b = subtract(this.b, db.multiply(learningRate));

        return propagatedError;

    }

    @Override
    public Matrix getWeight() {
        return this.w;
    }

    @Override
    public Matrix getBias() {
        return this.b;
    }

    @Override
    public int getInputDim() {
        return this.n;
    }

    @Override
    public int getOutputDim() {
        return this.m;
    }

    public static final void main(String[] args) {

        Matrix x = Matrix.fromArray(new double[][] {
            {3, 2, -1}, 
            {3, 2, -1}, 
            {3, 2, -1}, 
            {1, 0, 3}
        });

        Matrix y = Matrix.fromArray(new double[][] {
            {1, 3}, 
            {1, 3}, 
            {1, 3}, 
            {2, 1}
        });

        DenseLayer layer1 = new DenseLayer(3, 2, new SigmoidActivation());

        x.print("x");
        layer1.w.print("W");
        layer1.b.print("b");

        Matrix a1 = layer1.forwardPropagate(x);
        a1.print("a1");

        Matrix loss = subtract(a1, y);
        loss.print("loss");

        layer1.backwardPropagate(loss, 0.01).print("prop. err.");
        
        layer1.getWeight().print("W - updated");
        layer1.getBias().print("b - updated");


    }
    
}
