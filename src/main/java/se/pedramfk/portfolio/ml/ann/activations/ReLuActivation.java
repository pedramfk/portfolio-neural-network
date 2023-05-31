package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;


public final class ReLuActivation implements Activation {

    private static final ApplyValueFunction relu = (v) -> v <= 0.0 ? 0.0 : v;
    private static final ApplyValueFunction reluGrad = (v) -> v > 0.0 ? 1.0 : 0.0;

    @Override
    public Matrix getActivation(Matrix input) {
        return input.copy().apply(relu);
    }

    @Override
    public Matrix getActivationGradient(Matrix output) {
        return output.copy().apply(reluGrad);
    }
    
}
