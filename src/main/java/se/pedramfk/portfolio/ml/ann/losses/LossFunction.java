package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface LossFunction {

    public Matrix getCost(Matrix a, Matrix y);

    public Matrix getCostGradient(Matrix a, Matrix y);
    
}
