package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface Activation {

    /**
     * Retrieve activation a for layer output z.
     * 
     * @param z     linear output from layer
     * @return      activation
     */
    Matrix getActivation(Matrix z);

    /**
     * Retrieve derivative of activation function {@link #getActivation(Matrix)}.
     * 
     * @param z     linear output from layer
     * @return      derivative of activation
     */
    Matrix getActivationGradient(Matrix z);
    
}
