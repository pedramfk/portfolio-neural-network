package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;

public final class TanhActivation implements Activation {

    private static final ApplyValueFunction tanh = (v) -> Math.tanh(v);
    private static final ApplyValueFunction tanhGrad = (v) -> 1.0 - Math.pow(Math.tanh(v), 2);

    @Override
    public Matrix getActivation(Matrix input) {
        return input.copy().apply(tanh);
    }

    @Override
    public Matrix getActivationGradient(Matrix output) {
        return output.copy().apply(tanhGrad);
    }

}
