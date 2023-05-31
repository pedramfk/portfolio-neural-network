package se.pedramfk.portfolio.ml.metrics;

import se.pedramfk.portfolio.ml.utils.Matrix;

public final class ConfusionMatrix {

    final int tp, fp, tn, fn;

    final double tpr, fpr, tnr, fnr;

    public ConfusionMatrix(Matrix targets, Matrix predictions) {

        assert((targets.cols == 1) && (predictions.cols == 1));

        int truePositiveCount = 0;
        int falsePositiveCount = 0;
        int trueNegativeCount = 0;
        int falseNegativeCount = 0;

        for (int i = 0; i < targets.rows; i++) {

            int predictionLabel = predictions.get(i, 0) > .5 ? 1 : 0;
            int targetLabel = targets.get(i, 0) > .5 ? 1 : 0;

            truePositiveCount += predictionLabel * targetLabel;
            falsePositiveCount += predictionLabel * (1 - targetLabel);
            trueNegativeCount += (1 - predictionLabel) * (1 - targetLabel);
            falseNegativeCount += (1 - predictionLabel) * targetLabel;

        }

        this.tp = truePositiveCount;
        this.fp = falsePositiveCount;
        this.tn = trueNegativeCount;
        this.fn = falseNegativeCount;

        this.tpr = (double) truePositiveCount / (truePositiveCount + falseNegativeCount);
        this.fpr = (double) falsePositiveCount / (falsePositiveCount + truePositiveCount);
        this.tnr = (double) trueNegativeCount / (trueNegativeCount + falsePositiveCount);
        this.fnr = (double) falseNegativeCount / (truePositiveCount + falseNegativeCount);

    }


    @Override
    public String toString() {
        
        StringBuilder sb = new StringBuilder();

        final int boxWidth = 15;

        String tprString = String.format("%.1f %s", 100.0 * tpr, "%");
        String fnrString = String.format("%.1f %s", 100.0 * fnr, "%");
        String fprString = String.format("%.1f %s", 100.0 * fpr, "%");
        String tnrString = String.format("%.1f %s", 100.0 * tnr, "%");

        String tpcString = String.format("(%d)", tp);
        String fncString = String.format("(%d)", fn);
        String fpcString = String.format("(%d)", fp);
        String tncString = String.format("(%d)", tn);

        String tprBoxString = "";
        String fnrBoxString = "";
        String fprBoxString = "";
        String tnrBoxString = "";

        String tpcBoxString = "";
        String fncBoxString = "";
        String fpcBoxString = "";
        String tncBoxString = "";

        for (int i = 0; i < Math.floor((boxWidth - tprString.length()) / 2); i++ ) tprBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - fnrString.length()) / 2); i++ ) fnrBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - fprString.length()) / 2); i++ ) fprBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - tnrString.length()) / 2); i++ ) tnrBoxString += " ";

        for (int i = 0; i < Math.floor((boxWidth - tpcString.length()) / 2); i++ ) tpcBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - fncString.length()) / 2); i++ ) fncBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - fpcString.length()) / 2); i++ ) fpcBoxString += " ";
        for (int i = 0; i < Math.floor((boxWidth - tncString.length()) / 2); i++ ) tncBoxString += " ";

        tprBoxString += tprString;
        fnrBoxString += fnrString;
        fprBoxString += fprString;
        tnrBoxString += tnrString;

        tpcBoxString += tpcString;
        fncBoxString += fncString;
        fpcBoxString += fpcString;
        tncBoxString += tncString;

        for (int i = 0; i < (boxWidth - tprBoxString.length()); i++ ) tprBoxString += " ";
        for (int i = 0; i < (boxWidth - fnrBoxString.length()); i++ ) fnrBoxString += " ";
        for (int i = 0; i < (boxWidth - fprBoxString.length()); i++ ) fprBoxString += " ";
        for (int i = 0; i < (boxWidth - tnrBoxString.length()); i++ ) tnrBoxString += " ";

        for (int i = 0; i <= (boxWidth - tpcBoxString.length()); i++ ) tpcBoxString += " ";
        for (int i = 0; i <= (boxWidth - fncBoxString.length()); i++ ) fncBoxString += " ";
        for (int i = 0; i <= (boxWidth - fpcBoxString.length()); i++ ) fpcBoxString += " ";
        for (int i = 0; i <= (boxWidth - tncBoxString.length()); i++ ) tncBoxString += " ";
        sb.append("\n");
        sb.append("           Predicted     Predicted\n");
        sb.append("           True          False\n");
        sb.append("        ┌─────────────┬─────────────┐\n");
        sb.append(String.format("Actual  │%s│%s│\n", tprBoxString, fnrBoxString));
        sb.append(String.format("True    │%s│%s│\n", tpcBoxString, fncBoxString));
        sb.append("        ├─────────────┼─────────────┤\n");
        sb.append(String.format("Actual  │%s│%s│\n", fprBoxString, tnrBoxString));
        sb.append(String.format("False   │%s│%s│\n", fpcBoxString, tncBoxString));
        sb.append("        └─────────────┴─────────────┘\n");
        sb.append(String.format("\n· Accuracy: %.1f %s", 100.0 * (tp + tn) / (tp + fp + tn + fn), "%"));
        sb.append(String.format("\n· Precision: %.1f %s", 100.0 * tp / (tp + fp), "%"));
        sb.append(String.format("\n· Recall: %.1f %s", 100.0 * tp / (tp + fn), "%"));
        sb.append(String.format("\n· F1: %.1f %s", 100.0 * 2 * tp / (2 * tp + fp + fn), "%"));

        return sb.toString();
    }

    public void print() {
        System.out.println(toString());
    }
    
}
