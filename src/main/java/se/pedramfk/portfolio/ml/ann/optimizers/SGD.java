package se.pedramfk.portfolio.ml.ann.optimizers;

import se.pedramfk.portfolio.ml.ann.layers.Layer;
import se.pedramfk.portfolio.ml.utils.Matrix;


public final class SGD implements Optimizer {

    private final boolean decayLearningRate;
    private final double initLearningRate;
    private double currentLearningRate;

    private final boolean useMomentum;
    private final double momentum;

    public SGD(double lr, boolean lrDecay, double momentum, boolean useMomentum) {
        this.decayLearningRate = lrDecay;
        this.initLearningRate = lr;
        this.useMomentum = useMomentum;
        this.momentum = momentum;
    }

    @Override
    public double getCurrentLearningRate() {
        return this.currentLearningRate;
    }

    @Override
    public boolean decayLearningRate() {
        return this.decayLearningRate;
    }

    @Override
    public double getMomentum() {
        return this.momentum;
    }

    @Override
    public boolean useMomentum() {
        return this.useMomentum;
    }

    @Override
    public Matrix step(int iteration, Layer layer, Matrix gradient) {

        this.currentLearningRate = this.initLearningRate * (1 - Math.exp(-1.0 / iteration));

        return layer.backwardPropagate(gradient, this.currentLearningRate);
        
    }
    
}
