package se.pedramfk.portfolio.ml.ann.layers;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface Layer {
    
    Matrix forwardPropagate(Matrix input);

    Matrix backwardPropagate(Matrix error, double learningRate);

}
