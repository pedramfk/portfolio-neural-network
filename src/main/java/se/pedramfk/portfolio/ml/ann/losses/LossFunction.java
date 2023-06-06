package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;

public interface LossFunction {

    public double getLoss(Matrix yTrue, Matrix yHat);

    public Matrix getLossGradient(Matrix yTrue, Matrix yHat);
    
}
