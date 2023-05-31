package se.pedramfk.portfolio.ml.metrics;

import se.pedramfk.portfolio.ml.utils.Matrix;

public final class BinaryClassificationResults {

    private final ConfusionMatrix confusionMatrix;

    public BinaryClassificationResults(Matrix targets, Matrix predictions) {
        this.confusionMatrix = new ConfusionMatrix(targets, predictions);
    }

    public final ConfusionMatrix getConfusionMatrix() {
        return this.confusionMatrix;
    }

    public static final void main(String[] args) {

        //                TN -- TP -- FP -- FN -- FN -- TP -- TN -- TN -- FP
        double[][] y1 = {{.1}, {.8}, {.6}, {.1}, {.2}, {.9}, {.2}, {.2}, {.7}};
        double[][] y2 = {{0.}, {1.}, {.0}, {1.}, {1.}, {1.}, {.0}, {.0}, {1.}};

        Matrix pred = new Matrix(y1);
        Matrix target = new Matrix(y2);

        BinaryClassificationResults res = new BinaryClassificationResults(target, pred);

        System.out.println(res.getConfusionMatrix().toString());

    }
    
}
