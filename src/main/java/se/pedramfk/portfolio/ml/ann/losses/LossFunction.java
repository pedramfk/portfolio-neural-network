package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface LossFunction {

    public Matrix getLoss(Matrix a, Matrix y);

    public Matrix getLossGradient(Matrix a, Matrix y);
    
}
