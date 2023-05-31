package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface HiddenLayer extends Layer {

    Matrix getWeight();

    Matrix getBias();

    int getInputDim();

    int getOutputDim();
    
}
