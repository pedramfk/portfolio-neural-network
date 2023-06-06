package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;
import static se.pedramfk.portfolio.ml.utils.Matrix.*;

import se.pedramfk.portfolio.ml.ann.activations.Activation;
import se.pedramfk.portfolio.ml.ann.activations.SigmoidActivation;
import se.pedramfk.portfolio.ml.ann.losses.BinaryCrossEntropyLoss;


public final class DenseLayer implements Layer {

    private final int n, m;

    private Matrix w, b;

    private Matrix x, z;

    private final Activation activation;

    public DenseLayer(int n, int m, Activation activation) {
        
        this.n = n;
        this.m = m;

        this.w = new Matrix(m, n).initializeWithRandomValue(.1);
        this.b = new Matrix(m, 1).initializeWithRandomValue(.01);

        this.activation = activation;

    }


    @Override
    public Matrix forwardPropagate(Matrix input) {

        this.x = input.copy();

        this.z = transpose(add(multiply(this.w, transpose(input.copy())), repeat(this.b, 1, input.rows)));
        
        return activation.getActivation(this.z);

    }

    @Override
    public Matrix backwardPropagate(Matrix delta, double learningRate) {

        double lambda = .0001;

        Matrix activationGradient = dot(delta, activation.getActivationGradient(this.z));

        Matrix propagatedGradient = multiply(activationGradient, this.w);

        //Matrix weightGradient = multiply(transpose(activationGradient), this.x);
        Matrix weightGradient = add(multiply(transpose(activationGradient), this.x).multiply(1.0 / this.x.rows), multiply(this.w, 2 * lambda));

        this.w = subtract(this.w, weightGradient.multiply(learningRate));
        this.b = subtract(this.b, transpose(activationGradient).multiply(learningRate));

        return propagatedGradient;

    }

    @Override public Matrix getWeight() { return this.w; }

    @Override public Matrix getBias() { return this.b; }

    @Override public int getInputDim() { return this.n; }

    @Override public int getOutputDim() { return this.m; }

    public static final void main(String[] args) {

        final BinaryCrossEntropyLoss crossEntropyLoss = new BinaryCrossEntropyLoss();

        final double lr = 1e-3;

        Matrix x = fromArray(new double[][] {
            {.3, .2, .9, .2, .7}, 
            {.3, .4, .1, .0, .1}, 
            {.3, .2, .0, .1, .4}, 
            //{1, 0, 3}
        });

        Matrix y = fromArray(new double[][] {
            {1.0}, 
            {1.0}, 
            {0.0}, 
            //{0}
        });

        DenseLayer layer1 = new DenseLayer(5, 4, new SigmoidActivation());
        DenseLayer layer2 = new DenseLayer(4, 2, new SigmoidActivation());
        DenseLayer layer3 = new DenseLayer(2, 1, new SigmoidActivation());

        final Matrix layer1W0 = layer1.getWeight().copy();
        final Matrix layer2W0 = layer2.getWeight().copy();
        final Matrix layer3W0 = layer3.getWeight().copy();

        final Matrix yHat0 = layer3.forwardPropagate(layer2.forwardPropagate(layer1.forwardPropagate(x)));
        
        double loss0 = crossEntropyLoss.getLoss(y, yHat0);
        Matrix lossGrad0 = crossEntropyLoss.getLossGradient(y, yHat0);

        layer1.backwardPropagate(layer2.backwardPropagate(layer3.backwardPropagate(lossGrad0, lr), lr), lr);

        final Matrix layer1W1 = layer1.getWeight().copy();
        final Matrix layer2W1 = layer2.getWeight().copy();
        final Matrix layer3W1 = layer3.getWeight().copy();

        final Matrix yHat1 = layer3.forwardPropagate(layer2.forwardPropagate(layer1.forwardPropagate(x)));

        double loss1 = crossEntropyLoss.getLoss(y, yHat1);
        Matrix lossGrad1 = crossEntropyLoss.getLossGradient(y, yHat1);

        layer1.backwardPropagate(layer2.backwardPropagate(layer3.backwardPropagate(lossGrad1, lr), lr), lr);

        final Matrix layer1W2 = layer1.getWeight().copy();
        final Matrix layer2W2 = layer2.getWeight().copy();
        final Matrix layer3W2 = layer3.getWeight().copy();

        final Matrix yHat2 = layer3.forwardPropagate(layer2.forwardPropagate(layer1.forwardPropagate(x)));

        double loss2 = crossEntropyLoss.getLoss(y, yHat2);
        Matrix lossGrad2 = crossEntropyLoss.getLossGradient(y, yHat2);

        layer1W0.print("W1(0)", 5);
        layer1W1.print("W1(1)", 5);
        layer1W2.print("W1(2)", 5);

        layer2W0.print("W2(0)", 5);
        layer2W1.print("W2(1)", 5);
        layer2W2.print("W2(2)", 5);

        layer3W0.print("W3(0)", 5);
        layer3W1.print("W3(1)", 5);
        layer3W2.print("W3(2)", 5);

        System.out.println("L(0): " + loss0);
        System.out.println("L(1): " + loss1);
        System.out.println("L(2): " + loss2);

        lossGrad0.print("L'(0)", 4);
        lossGrad1.print("L'(1)", 4);
        lossGrad2.print("L'(2)", 4);


    }
    
}
