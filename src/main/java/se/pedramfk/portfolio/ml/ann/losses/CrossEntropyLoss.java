package se.pedramfk.portfolio.ml.ann.losses;

import se.pedramfk.portfolio.ml.utils.Matrix;


public final class CrossEntropyLoss implements LossFunction {

    @Override
    public Matrix getCost(Matrix a, Matrix y) {

        Matrix batchLoss = new Matrix(1, y.cols);

        //Matrix l = Matrix.sum(Matrix.multiply(y, a), 0);

        for (int col = 0; col < y.cols; col++) {
            double sum = 0.0;
            for (int row = 0; row < y.rows; row++) {
                double prediction = a.get(row, col);
                double target = y.get(row, col);
                sum += target * Math.log(prediction);
                //double v = Math.log(l.get(i, j) == 0.0 ? Float.MIN_VALUE : l.get(i, j));
            }
            batchLoss.set(0, col, sum == 0.0 ? -Float.MIN_VALUE : -sum);
        }
        
        //return Matrix.mean(l, 1);
        return batchLoss;
        
    }

    @Override
    public Matrix getCostGradient(Matrix a, Matrix y) {

        Matrix g = new Matrix(a.rows, a.cols);
        for (int row = 0; row < y.rows; row++) {
            for (int col = 0; col < y.cols; col++) {
                double prediction = a.get(row, col);
                double target = y.get(row, col);
                double v = - (target / prediction) + (1 - target) / (1 - prediction);
                g.set(row, col, v);
            }
            
        }
        return g;
    }
    
}
