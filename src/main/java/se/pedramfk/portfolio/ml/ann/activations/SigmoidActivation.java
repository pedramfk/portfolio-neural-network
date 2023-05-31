package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;


public final class SigmoidActivation implements Activation {

    private static final ApplyValueFunction sigmoid = (v) -> 1.0 / (1.0 + Math.exp(-v));
    private static final ApplyValueFunction sigmoidGrad = (v) -> sigmoid.apply(v) * (1. - sigmoid.apply(v));

    @Override
    public Matrix getActivation(Matrix input) {
        return input.copy().apply(sigmoid);
    }

    @Override
    public Matrix getActivationGradient(Matrix output) {
        return output.copy().apply(sigmoidGrad);
    }
    
}
