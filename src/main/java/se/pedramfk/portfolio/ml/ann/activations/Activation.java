package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface Activation {

    Matrix getActivation(Matrix input);

    Matrix getActivationGradient(Matrix output);
    
}
