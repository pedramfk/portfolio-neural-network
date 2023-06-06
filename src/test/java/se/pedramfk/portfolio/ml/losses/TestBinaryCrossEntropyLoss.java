package se.pedramfk.portfolio.ml.losses;

import se.pedramfk.portfolio.ml.ann.losses.BinaryCrossEntropyLoss;
import se.pedramfk.portfolio.ml.utils.Matrix;

public class TestBinaryCrossEntropyLoss {

    static final BinaryCrossEntropyLoss lossFunction = new BinaryCrossEntropyLoss();


    public static final void main(String[] args) {

        double[][] yTargetArray = new double[][] {{1, 0, 0, 1, 1}};
        double[][] yPredArray = new double[][] {{.9, .2, .7, .2, .0}};

        Matrix yTarget = new Matrix(yTargetArray);
        Matrix yPred = new Matrix(yPredArray);

        yTarget.label = "yTarget";
        yPred.label = "yPred";

        double loss = lossFunction.getLoss(yTarget, yPred);
        Matrix lossGradient = lossFunction.getLossGradient(yTarget, yPred);

        lossGradient.label = "dL";

        yTarget.print();
        yPred.print();

        System.out.println("L(a, y) = " + loss);
        lossGradient.print();

    }
    
}
