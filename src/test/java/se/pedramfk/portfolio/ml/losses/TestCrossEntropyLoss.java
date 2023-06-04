package se.pedramfk.portfolio.ml.losses;

import se.pedramfk.portfolio.ml.ann.losses.CrossEntropyLoss;
import se.pedramfk.portfolio.ml.utils.Matrix;

public class TestCrossEntropyLoss {


    public static final void main(String[] args) {

        //double[][] yTargetArray = new double[][] {{1, 0, 0, 1, 1}};
        //double[][] yPredArray = new double[][] {{.9, .2, .7, .2, .0}};

        double[][] yTargetArray = new double[][] {{1, 0, 0, 0}};
        double[][] yPredArray = new double[][] {{.775, .116, .039, .070}};

        Matrix yTarget = new Matrix(yTargetArray);
        Matrix yPred = new Matrix(yPredArray);

        yTarget.label = "yTarget";
        yPred.label = "yPred";

        CrossEntropyLoss l = new CrossEntropyLoss();

        Matrix cost = l.getLoss(yPred, yTarget);
        Matrix costGradient = l.getLossGradient(yPred, yTarget);

        cost.label = "L(a, y)";
        costGradient.label = "dL(a, y)";

        yTarget.print();
        yPred.print();

        cost.print("L(a, y)", 5);
        costGradient.print("dL(a, y)", 3);

    }
    
}
