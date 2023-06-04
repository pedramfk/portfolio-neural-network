package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;
import static se.pedramfk.portfolio.ml.utils.Matrix.*;

import se.pedramfk.portfolio.ml.ann.activations.Activation;
import se.pedramfk.portfolio.ml.ann.activations.SigmoidActivation;
import se.pedramfk.portfolio.ml.ann.losses.CrossEntropyLoss;


public final class DenseLayer implements HiddenLayer {

    private final int n, m;

    private Matrix w, b;

    private Matrix x, z;

    private final Activation activationLayer;

    public DenseLayer(int n, int m, Activation activationLayer) {
        
        this.n = n;
        this.m = m;

        this.w = new Matrix(m, n).initializeWithRandomValue(.01);
        this.b = new Matrix(m, 1).initializeWithRandomValue(.001);

        this.activationLayer = activationLayer;

    }


    @Override
    public Matrix forwardPropagate(Matrix input) {

        this.x = input.copy();
        this.z = add(multiply(this.w, this.x), repeat(this.b, 1, input.cols));
        return activationLayer.getActivation(this.z);

    }

    @Override
    public Matrix backwardPropagate(Matrix delta, double learningRate) {

        Matrix aGrad = activationLayer.getActivationGradient(this.z);
        
        Matrix forwardDelta = multiply(transpose(this.w), delta);
        Matrix currentDelta = dot(delta, aGrad);

        Matrix wGrad = multiply(this.x, transpose(currentDelta)).multiply(learningRate);
        Matrix bGrad = currentDelta.multiply(learningRate);

        this.w = Matrix.subtract(this.w, transpose(wGrad));
        this.b = Matrix.subtract(this.b, bGrad);
        
        return forwardDelta;

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

        double[][] xArray = new double[][] {
            {3, 2, -1}, 
            {3, 2, -1}, 
            {3, 2, -1}, 
            {1, 0, 3}
        };

        double[][] yArray = new double[][] {
            {1}, 
            {1}, 
            {1}, 
            {0}
        };

        Matrix x = transpose(fromArray(xArray));
        Matrix y = transpose(fromArray(yArray));

        DenseLayer layer1 = new DenseLayer(3, 2, new SigmoidActivation());
        DenseLayer layer2 = new DenseLayer(2, 1, new SigmoidActivation());

        layer1.getWeight().print("Layer 1 - W - Iteration 0", 3);
        layer2.getWeight().print("Layer 2 - W - Iteration 0", 3);

        layer1.getBias().print("Layer 1 - b - Iteration 0", 3);
        layer2.getBias().print("Layer 2 - b - Iteration 0", 3);

        Matrix a1 = layer1.forwardPropagate(x);
        Matrix a2 = layer2.forwardPropagate(a1);
        
        a1.print("a1", 3);
        a2.print("a2", 3);

        final CrossEntropyLoss crossEntropyLoss = new CrossEntropyLoss();

        Matrix loss = crossEntropyLoss.getLoss(a2, y);
        Matrix lossGrad = crossEntropyLoss.getLossGradient(a2, y);

        loss.print("Cross-Entropy Loss - Iteration 0", 3);
        lossGrad.print("Cross-Entropy Loss Gradient - Iteration 0", 3);

        Matrix delta2 = layer2.backwardPropagate(lossGrad, 0.01);
        Matrix delta1 = layer1.backwardPropagate(delta2, 0.01);

        delta2.print("layer 2 - delta", 3);
        delta1.print("layer 1 - delta", 3);

        layer1.getWeight().print("Layer 1 - W - Iteration 1", 3);
        layer2.getWeight().print("Layer 2 - W - Iteration 1", 3);

        layer1.getBias().print("Layer 1 - b - Iteration 1", 3);
        layer2.getBias().print("Layer 2 - b - Iteration 1", 3);

        Matrix a2New = layer2.forwardPropagate(layer1.forwardPropagate(x));
        Matrix updatedLoss = crossEntropyLoss.getLoss(a2New, y);
        Matrix updatedLossGrad = crossEntropyLoss.getLossGradient(a2New, y);

        updatedLoss.print("Cross-Entropy Loss - Iteration 1", 3);
        updatedLossGrad.print("Cross-Entropy Loss Gradient - Iteration 1", 3);

    }
    
}
