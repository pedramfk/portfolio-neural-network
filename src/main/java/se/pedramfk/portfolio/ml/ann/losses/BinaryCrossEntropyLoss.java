package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;


public final class BinaryCrossEntropyLoss implements LossFunction {

    private static final ApplyValueFunction log = (v) -> - Math.log(v <= 0.0 ? Float.MIN_NORMAL : v);

    @Override
    public double getLoss(Matrix yTrue, Matrix yHat) {
        return Matrix.sum(Matrix.dot(yTrue, Matrix.apply(yHat, log)));
    }

    @Override
    public Matrix getLossGradient(Matrix yTrue, Matrix yHat) {

        Matrix g = new Matrix(yTrue.rows, yTrue.cols);

        for (int col = 0; col < yTrue.cols; col++) {
            
            for (int row = 0; row < yTrue.rows; row++) {
                double target = yTrue.get(row, col);
                double prediction = yHat.get(row, col);
                double predClip = prediction == 0.0 ? 1e-12 : (prediction == 1.0 ? prediction - 1e-12 : prediction);
                double v = - (target / predClip) + (1 - target) / (1 - predClip);
                g.set(row, col, v);
            }
            
        }

        //return g;
        return Matrix.mean(g, 0);

        /*
         assert(yTrue.rows == yHat.rows);
        assert((yTrue.cols == 1) && (yHat.cols == 1));

        final int n = yTrue.rows;

        Matrix lossGradient = new Matrix(n, 1);

        for (int i = 0; i < n; i++) {

            double target = yTrue.get(i, 0);
            double pred = yHat.get(i, 0);

            double predClip = pred <= EPSILON ? EPSILON : (pred >= 1.0 - EPSILON ? 1.0 - EPSILON : pred);

            lossGradient.set(i, 0, ((target / predClip) + (1.0 - target) / (1.0 - predClip)));

        }

        return Matrix.sum(lossGradient, 0).multiply(-1);
         */
        

    }
    
}
