package se.pedramfk.portfolio.ml.utils;

import java.io.BufferedReader;
import java.io.FileReader;


public final class MatrixData {

    public static final double getMaxValue(double[][] data, int col) {
        double maxValue = - Double.MAX_VALUE;
        for (int r = 0; r < data.length; r++) {
            maxValue = data[r][col] > maxValue ? data[r][col] : maxValue;
        }
        return maxValue;
    }

    public static final double[] getMaxValues(double[][] x) {
        double[] maxValues = new double[x[0].length];
        for (int c = 0; c < x[0].length; c++) {
            maxValues[c] = getMaxValue(x, c);
        }
        return maxValues;
    }

    public static final RowAndColCount getRowAndColCount(String path, String sep) throws Exception {

        int r = 0;
        int c = 0;
        BufferedReader br = null;

        try {
            br = new BufferedReader(new FileReader(path));
            String line;
            while ((line = br.readLine()) != null) {
                if (c == 0) c = line.split(sep).length - 1;
                r++;
            }
        } catch (Exception e) {
            throw e;
        } finally {
            try {
                br.close();
            } catch (Exception e) {
            }
        }

        return new RowAndColCount(r, c);
        
    }

    public static final int[][] getRandomMatrixIndices(Matrix matrix, int n) {
        
        java.util.List<Integer> rowIndices = new java.util.ArrayList<>();
        java.util.List<Integer> colIndices = new java.util.ArrayList<>();

        int[][] indices = new int[n][2];

        for (int i = 0; i < n; i++) {

            int v1 = (int) Math.floor(Math.random() * matrix.rows);
            int v2 = (int) Math.floor(Math.random() * matrix.cols);

            while (rowIndices.contains(v1) && colIndices.contains(v2) ) {
                v1 = (int) Math.floor(Math.random() * matrix.rows);
                v2 = (int) Math.floor(Math.random() * matrix.cols);
            }

            rowIndices.add(v1);
            colIndices.add(v2);

            indices[i][0] = v1;
            indices[i][1] = v2;

        }
        
        return indices;
        
    }

    public static final Integer[] getRandomIndices(int n) {
        
        java.util.List<Integer> setIndices = new java.util.ArrayList<>();

        for (int i = 0; i < n; i++) {
            int v;
            while (setIndices.contains( v = (int) Math.floor(Math.random() * n) ));
            setIndices.add(v);
        }
        
        return setIndices.toArray(new Integer[n]);

    }
    
    public static final TrainAndTestData loadTrainAndTestData(String path, String sep, double trainSplitSize) throws Exception {

        final InputAndOutputData inputAndOutputData = loadInputAndOutputData(path, sep);

        final int total = inputAndOutputData.x.length;
        final int trainSize = (int) Math.floor(trainSplitSize * total);
        final int testSize = total - trainSize;

        final double[] maxValues = getMaxValues(inputAndOutputData.getX());
        
        final double[][] xTrain = new double[trainSize][inputAndOutputData.x[0].length];
        final double[][] yTrain = new double[trainSize][1];

        final double[][] xTest = new double[testSize][inputAndOutputData.x[0].length];
        final double[][] yTest = new double[testSize][1];

        final Integer[] randomIndices = getRandomIndices(total);
        
        for (int i = 0; i < trainSize; i++) {
            xTrain[i] = inputAndOutputData.x[randomIndices[i]];
            yTrain[i] = inputAndOutputData.y[randomIndices[i]];
        }

        for (int i = 0; i < testSize; i++) {
            xTest[i] = inputAndOutputData.x[randomIndices[i + trainSize]];
            yTest[i] = inputAndOutputData.y[randomIndices[i + trainSize]];
        }

        return new TrainAndTestData(new InputAndOutputData(xTrain, yTrain, maxValues), new InputAndOutputData(xTest, yTest, maxValues));

    }

    public static final InputAndOutputData loadInputAndOutputData(String path, String sep) throws Exception {

        final RowAndColCount rowAndColCount = getRowAndColCount(path, sep);

        double[][] x = new double[rowAndColCount.nRows][rowAndColCount.nCols];
        double[][] y = new double[rowAndColCount.nRows][1];
        
        BufferedReader br = null;

        try {

            br = new BufferedReader(new FileReader(path));

            int r = 0;
            String line;
            while ((line = br.readLine()) != null) {
                String[] rawData = line.split(sep);
                for (int i = 0; i < x[0].length; i++) {
                    x[r][i] = Double.parseDouble(rawData[i]);
                }
                y[r][0] = Double.parseDouble(rawData[x[0].length]);
                r++;
            }

        } catch (Exception e) {

            throw e;

        } finally {

            try {
                br.close();
            } catch (Exception e) {

            }

        }

        return new InputAndOutputData(x, y);
        
    }


    public static final class RowAndColCount {

        final int nRows, nCols;

        RowAndColCount(int nRows, int nCols) {
            this.nRows = nRows;
            this.nCols = nCols;
        }
        
    }

    public static final class InputAndOutputData {

        private final double[][] x, y;
        private double[] maxValues;

        InputAndOutputData(double[][] x, double[][] y) {
            this.x = x;
            this.y = y;
            this.maxValues = getMaxValues(x);
        }

        InputAndOutputData(double[][] x, double[][] y, double[] maxValues) {
            this.x = x;
            this.y = y;
            this.maxValues = maxValues;
        }

        public final double[][] getX() {
            return x;
        }

        public final double[][] getY() {
            return y;
        }

        public final void setMaxValues(double[] maxValues) {
            this.maxValues = maxValues;
        }

        public final double[][] getNormalizedX() {
            double[][] xNorm = new double[x.length][x[0].length];
            for (int c = 0; c < x[0].length; c++) {
                for (int r = 0; r < x.length; r++) {
                    xNorm[r][c] = x[r][c] / maxValues[c];
                }
            }
            return xNorm;
        }

        public final double[][] getNormalizedX(double[] maxVals) {
            double[][] xNorm = new double[x.length][x[0].length];
            for (int c = 0; c < x[0].length; c++) {
                for (int r = 0; r < x.length; r++) {
                    xNorm[r][c] = x[r][c] / maxVals[c];
                }
            }
            return xNorm;
        }

    }

    public static final class TrainAndTestData {

        private final InputAndOutputData train, test;

        public TrainAndTestData(InputAndOutputData train, InputAndOutputData test) {
            this.train = train;
            this.test = test;
        }

        public final InputAndOutputData getTrainData() {
            return train;
        }

        public final InputAndOutputData getTestData() {
            return test;
        }

    }


    public static final void main(String[] args) throws Exception {

        final String path = "/Users/pedramfk/workspace/git/portfolio/neural-network/src/test/resources/pima-indians-diabetes.csv";

        final TrainAndTestData trainAndTestData = loadTrainAndTestData(path, ",", .8);

        Matrix xTrain = Matrix.fromArray(trainAndTestData.getTrainData().getNormalizedX());
        Matrix yTrain = Matrix.fromArray(trainAndTestData.getTrainData().getY());

        Matrix xTest = Matrix.fromArray(trainAndTestData.getTestData().getNormalizedX());
        Matrix yTest = Matrix.fromArray(trainAndTestData.getTestData().getY());

        //xTrain.print("xTrain");
        //yTrain.print("yTrain");

        System.out.println(xTrain.rows + " x " + xTrain.cols);
        System.out.println(yTrain.rows + " x " + yTrain.cols);

        System.out.println(xTest.rows + " x " + xTest.cols);
        System.out.println(yTest.rows + " x " + yTest.cols);

    }
    
}
