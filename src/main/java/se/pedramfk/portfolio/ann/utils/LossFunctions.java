package se.pedramfk.portfolio.ann.utils;


public final class LossFunctions {

    public static final double getMSE(MatrixData yTrue, MatrixData yPred) {

        MatrixData yDiff = MatrixData.subtract(yTrue, yPred);
        MatrixData yDiffTransponsed = MatrixData.transpose(yDiff);
        return MatrixData.multiply(yDiffTransponsed, yDiff).divide(yTrue.nRows).get(0, 0);

    }

    public static final MatrixData getDeltaMSE(MatrixData yTrue, MatrixData yPred) {

        return MatrixData.subtract(yTrue, yPred).multiply(-2).divide(yTrue.nRows);

        /*
        double[] res = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            res[i] = 2 * (predicted[i] - actual[i]) / actual.length;
        }

        return res;
         */
        
    }

    public static final MatrixData getDeltaMSE(MatrixData error) {

        return error.copy().multiply(-2).divide(error.nRows);

        /*
        double[] res = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            res[i] = 2 * (predicted[i] - actual[i]) / actual.length;
        }

        return res;
         */
        
    }

    public static final double getAccuracy(MatrixData yTrue, MatrixData yPred) {

        int n = yTrue.nRows;
        double correct = 0;

        for (int i = 0; i < n; i++) {
            double pred = yPred.get(i, 0) > .5 ? 1.0 : 0.0;
            double actual = yTrue.get(i, 0) > .5 ? 1.0 : 0.0;
            if (pred == actual) correct++;
        }

        return correct / n;

    }
    
}
