package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;
import static se.pedramfk.portfolio.ml.utils.Matrix.*;

import se.pedramfk.portfolio.ml.ann.activations.Activation;
import se.pedramfk.portfolio.ml.ann.activations.SigmoidActivation;
import se.pedramfk.portfolio.ml.ann.losses.CrossEntropyLoss;


public final class DenseLayer implements HiddenLayer {

    private final int n, m;

    private Matrix w, b;

    private Matrix x, z, a;

    private final Activation activationLayer;

    public DenseLayer(int n, int m, Activation activationLayer) {
        
        this.n = n;
        this.m = m;

        this.w = new Matrix(m, n).initializeWithRandomValue(.1);
        this.b = new Matrix(m, 1).initializeWithRandomValue(.1);

        this.activationLayer = activationLayer;

    }


    @Override
    public Matrix forwardPropagate(Matrix input) {

        this.x = input.copy();
        this.z = add(multiply(this.w, this.x), repeat(this.b, 1, input.cols));
        this.a = activationLayer.getActivation(this.z);

        return this.a;

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
        
        //this.w.subtract(transpose(wGrad));
        //this.b.subtract(bGrad);
        
        return forwardDelta;

        //Matrix dZ = multiply(dY, transpose(this.w));
        //Matrix dC = multiply(dZ, dA);        
        //Matrix dW = multiply(transpose(this.x), dC);
        //Matrix db = mean(transpose(dC), 0);
        //this.w = subtract(this.w, dW.multiply(learningRate / this.x.rows));
        //this.b = subtract(this.b, db.multiply(learningRate / this.x.rows));
        //return dC;

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

        Matrix a1 = layer1.forwardPropagate(x);
        Matrix a2 = layer2.forwardPropagate(a1);

        Matrix layer1W = layer1.getWeight();
        Matrix layer1b = layer1.getBias();

        Matrix layer2W = layer2.getWeight();
        Matrix layer2b = layer2.getBias();
        
        a1.print("a1");
        a2.print("a2");

        //Matrix loss = subtract(a2, y);
        Matrix loss = new CrossEntropyLoss().getCost(a2, y);

        subtract(a2, y).print("loss - quadratic");;
        loss.print("loss - cross-entropy");
        Matrix lossGrad = new CrossEntropyLoss().getCostGradient(a2, y);
        lossGrad.print("loss grad - cross-entropy");

        Matrix delta2 = layer2.backwardPropagate(lossGrad, 0.01);
        Matrix delta1 = layer1.backwardPropagate(delta2, 0.01);

        delta2.print("layer 2 - delta");
        delta1.print("layer 1 - delta");

        layer1W.print("Layer 1 W - Pre");
        layer1.getWeight().print("Layer 1 W - Post");

        layer1b.print("Layer 1 b - Pre");
        layer1.getBias().print("Layer 1 b - Post");

        layer2W.print("Layer 2 W - Pre");
        layer2.getWeight().print("Layer 2 W - Post");

        layer2b.print("Layer 2 b - Pre");
        layer2.getBias().print("Layer 2 b - Post");

        Matrix updatedLoss = layer2.forwardPropagate(layer1.forwardPropagate(x));
        loss.print("loss - cross-entropy - Pre");
        updatedLoss.print("loss - cross-entropy - Post");

    }
    
}
