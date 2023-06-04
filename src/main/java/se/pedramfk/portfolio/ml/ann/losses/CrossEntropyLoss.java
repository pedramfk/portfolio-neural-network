package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;

import static se.pedramfk.portfolio.ml.utils.Matrix.dot;
import static se.pedramfk.portfolio.ml.utils.Matrix.mean;
import static se.pedramfk.portfolio.ml.utils.Matrix.apply;

public final class CrossEntropyLoss implements LossFunction {

    private static final ApplyValueFunction log = (v) -> - Math.log(v <= 0.0 ? Float.MIN_NORMAL : v);

    @Override
    public Matrix getLoss(Matrix a, Matrix y) {
        return mean(dot(y, apply(a, log)), 1).multiply(-1.0);
    }

    @Override
    public Matrix getLossGradient(Matrix a, Matrix y) {

        Matrix g = new Matrix(a.rows, a.cols);

        for (int col = 0; col < a.cols; col++) {
            for (int row = 0; row < a.rows; row++) {

                double target = y.get(row, col);
                double prediction = a.get(row, col) == 0.0 ? Float.MIN_NORMAL : a.get(row, col);
                double v = - (target / prediction) + (1 - target) / (1 - prediction);
                g.set(row, col, v);
            }
            
        }

        return g;
        //return Matrix.mean(g, 1);



        //return Matrix.subtract(a, y);

    }
    
}
