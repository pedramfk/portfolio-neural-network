package se.pedramfk.portfolio.ml.ann.results;

public class ModelResults {

    private final double[] trainAccuracy;
    private final double[] valAccuracy;

    private final double[] trainCost;
    private final double[] valCost;

    private final double[] trainLoss;
    private final double[] valLoss;

    public ModelResults(int epochs) {

        this.trainAccuracy = new double[epochs];
        this.valAccuracy = new double[epochs];

        this.trainCost = new double[epochs];
        this.valCost = new double[epochs];

        this.trainLoss = new double[epochs];
        this.valLoss = new double[epochs];

    }

    public void addAccuracy(int epoch, double trainResult, double valResult) {
        this.trainAccuracy[epoch] = trainResult;
        this.valAccuracy[epoch] = valResult;
    }

    public void addCost(int epoch, double trainResult, double valResult) {
        this.trainCost[epoch] = trainResult;
        this.valCost[epoch] = valResult;
    }

    public void addLoss(int epoch, double trainResult, double valResult) {
        this.trainLoss[epoch] = trainResult;
        this.valLoss[epoch] = valResult;
    }

    public double[] getTrainAccuracy() {
        return this.trainAccuracy;
    }

    public double[] getValAccuracy() {
        return this.valAccuracy;
    }

    public double[] getTrainCost() {
        return this.trainCost;
    }

    public double[] getValCost() {
        return this.valCost;
    }

    public double[] getTrainLoss() {
        return this.trainLoss;
    }

    public double[] getValLoss() {
        return this.valLoss;
    }
    
}
