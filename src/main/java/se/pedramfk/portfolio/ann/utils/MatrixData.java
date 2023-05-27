package se.pedramfk.portfolio.ann.utils;

import java.util.function.Function;

public final class MatrixData {

    private final double[][] arrayData;

    public final int nRows, nCols;

    public MatrixData(double[][] arrayData) {
        this.arrayData = copyArray(arrayData);
        this.nRows = this.arrayData.length;
        this.nCols = this.arrayData[0].length;
    }

    public MatrixData(MatrixData matrixData) { this(copyArray(matrixData.get())); }
    public MatrixData(double[] arrayData) { this(copyArray(arrayData)); }
    public MatrixData(int rows, int cols) { this(new double[rows][cols]); }

    public final MatrixData initializeWithValue(double value) {

        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] = value;
            }
        }

        return this;

    }

    public final MatrixData initializeWithRandomValue() {

        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] = Math.random() - .5;
            }
        }

        return this;

    }

    public final MatrixData copy() { return new MatrixData(arrayData); }
    public static final MatrixData create(double[] arrayData) { return new MatrixData(arrayData); }
    public static final MatrixData create(double[][] arrayData) { return new MatrixData(arrayData); }

    public final double[][] get() { return this.arrayData; }
    public final double[] get(int row) { return this.arrayData[row]; }
    public final double get(int row, int col) { return this.arrayData[row][col]; }

    public final void set(int row, int col, double v) { this.arrayData[row][col] = v; }
    public final void set(int row, double[] v) { this.arrayData[row] = v; }

    public final MatrixData add(double value) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] += value;
            }
        }
        return this;
    }

    public final MatrixData add(MatrixData matrixData) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] += matrixData.get(i, j);
            }
        }
        return this;
    }

    public final MatrixData subtract(double value) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] -= value;
            }
        }
        return this;
    }

    public final MatrixData subtract(MatrixData matrixData) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] -= matrixData.get(i, j);
            }
        }
        return this;
    }

    public final MatrixData multiply(double value) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] *= value;
            }
        }
        return this;
    }

    public final MatrixData multiply(MatrixData matrixData) {
        return multiply(this, matrixData);
    }

    public final MatrixData divide(double value) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] /= value;
            }
        }
        return this;
    }

    public static final MatrixData add(MatrixData a, MatrixData b) {

        final MatrixData res = new MatrixData(a.nRows, a.nCols);

        for (int i = 0; i < res.nRows; i++) {
            for (int j = 0; j < res.nCols; j++) {
                double value = a.get(i, j) + b.get(i, j);
                res.set(i, j, value);
            }
        }

        return res;

    }

    public static final MatrixData subtract(MatrixData a, MatrixData b) {
        final MatrixData res = new MatrixData(a.nRows, a.nCols);
        for (int i = 0; i < res.nRows; i++) {
            for (int j = 0; j < res.nCols; j++) {
                double value = a.get(i, j) - b.get(i, j);
                res.set(i, j, value);
            }
        }
        return res;
    }

    public static final MatrixData dot(MatrixData a, MatrixData b) {
        final MatrixData res = new MatrixData(a.nRows, a.nCols);
        for (int i = 0; i < res.nRows; i++) {
            for (int j = 0; j < res.nCols; j++) {
                double value = a.get(i, j) * b.get(i, j);
                res.set(i, j, value);
            }
        }
        return res;
    }

    public static final MatrixData multiply(MatrixData a, MatrixData b) {

        final MatrixData res = new MatrixData(a.nRows, b.nCols);

        for (int i = 0; i < res.nRows; i++) {  
            for (int j = 0; j < res.nCols; j++) {  

                double value = 0.0;
                for (int k = 0; k < a.nCols; k++) { 
                    value += a.get(i, k) * b.get(k, j);
                }
                res.set(i, j, value);

            }
        }

        return res;

    }

    public static final MatrixData transpose(MatrixData matrixData) {
        final MatrixData res = new MatrixData(matrixData.nCols, matrixData.nRows);
        for (int i = 0; i < res.nRows; i++) {
            for (int j = 0; j < res.nCols; j++) {
                res.set(i, j, matrixData.get(j, i));
            }
        }
        return res;
    }

    public final MatrixData transpose() {
        final MatrixData res = new MatrixData(this.nCols, this.nRows);
        for (int i = 0; i < res.nRows; i++) {
            for (int j = 0; j < res.nCols; j++) {
                res.set(i, j, get(j, i));
            }
        }
        return res;
    }

    public final MatrixData apply(Function<Double, Double> function) {
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                this.arrayData[i][j] = function.apply(this.arrayData[i][j]);
            }
        }
        return this;
    }

    public void print() {
        System.out.println();
        for (int i = 0; i < this.nRows; i++) {
            for (int j = 0; j < this.nCols; j++) {
                System.out.print(get(i, j) + "\t");
            }
            System.out.println();
        }
    }

    public void print(String name) {
        System.out.println();
        System.out.print(name);
        print();
    }

    private static final double[][] copyArray(double[][] arrayData) {
        final double[][] arrayDataCopy = new double[arrayData.length][arrayData[0].length];
        for (int i = 0; i < arrayData.length; i++) {
            for (int j = 0; j < arrayData[0].length; j++) {
                arrayDataCopy[i][j] = arrayData[i][j];
            }
        }
        return arrayDataCopy;
    }

    private static final double[][] copyArray(double[] arrayData) {
        return copyArray(new double[][] {arrayData});
    }

    public static final void main(String[] args) {

        MatrixData a = MatrixData.create(
            new double[][] {
                new double[] {3, 2}
            });

        MatrixData b = MatrixData.create(
            new double[][] {
                new double[] {2, 1}
            });

        a.print();
        b.print();

        MatrixData c = MatrixData.multiply(a.transpose(), b);
        MatrixData d = MatrixData.transpose(c);

        c.print();
        d.print();

    }
    
}
