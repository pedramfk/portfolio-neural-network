package se.pedramfk.portfolio.ml.utils;

import se.pedramfk.portfolio.ml.utils.ApplyFunctions.*;
import static se.pedramfk.portfolio.ml.utils.MatrixFunctions.*;


public final class Matrix {

    private double[][] arrayData;

    public final int rows, cols;

    public String label = "Matrix";

    public Matrix(double[][] arrayData) {
        this.arrayData = copyArray(arrayData);
        this.rows = arrayData.length;
        this.cols = arrayData[0].length;
    }

    public Matrix(int rows, int cols) {
        this(new double[rows][cols]);
    }

    public Matrix(Matrix matrix) {
        this(copyArray(matrix.arrayData));
    }

    /**
     * <p>Applies output from provided function to each entry.</p>
     * 
     * @param function      function to be applied
     * @return mutated instance of Matrix object
     */
    private final Matrix iterator(ApplyVoidFunction function) {
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                this.arrayData[row][col] = function.apply();
            }
        }
        return this;        
    }

    /**
     * <p>Applies output from provided function to each entry.</p>
     * <p><b>Note:</b> Assumes that function takes matrix entry as input.</p>
     * 
     * @param function      function to be applied
     * @return mutated instance of Matrix object
     */
    private final Matrix iterator(ApplyValueFunction function) {
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                this.arrayData[row][col] = function.apply(this.arrayData[row][col]);
            }
        }
        return this;
    }

    /**
     * <p>Applies output from provided function to each entry.</p>
     * <p><b>Note:</b> Assumes that function takes provided value as input.</p>
     * 
     * @param function      function to be applied
     * @param value         value to be provided when calling function
     * @return mutated instance of Matrix object
     */
    private final Matrix iterator(ApplyValueFunction function, double value) {
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                this.arrayData[row][col] = function.apply(value);
            }
        }
        return this;
    }

    /**
     * <p>Applies output from provided function to each entry.</p>
     * <p><b>Note:</b> Assumes that function takes matrix entry and provided value as input.</p>
     * 
     * @param function      function to be applied
     * @param value         value to be provided when calling function
     * @return mutated instance of Matrix object
     */
    private final Matrix iterator(ApplyValuesFunction function, double value) {
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                this.arrayData[row][col] = function.apply(this.arrayData[row][col], value);
            }
        }
        return this;
    }

    /**
     * <p>Applies output from provided function to each entry.</p>
     * <p><b>Note:</b> Assumes that function takes matrix entry and provided value as input.</p>
     * 
     * @param function      function to be applied
     * @param value         value to be provided when calling function
     * @return mutated instance of Matrix object
     */
    private final Matrix iterator(ApplyValuesFunction function, Matrix matrix) {
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                this.arrayData[row][col] = function.apply(this.arrayData[row][col], matrix.arrayData[row][col]);
            }
        }
        return this;
    }

    public final Matrix initializeWithValue(double value) {
        return iterator(initWithValue, value);
    }

    public final Matrix initializeWithRandomValue() {
        return iterator(initWithRandom);
    }

    public final Matrix initializeWithRandomValue(double scale) {
        return iterator(initWithRandomScaling, scale);
    }

    public final Matrix add(double v) {
        return iterator(addValues, v);
    }

    public final Matrix add(Matrix matrix) {
        return iterator(addValues, matrix);
    }

    public final Matrix subtract(double v) {
        return iterator(subtractValues, v);
    }

    public final Matrix subtract(Matrix matrix) {
        return iterator(subtractValues, matrix);
    }

    public final Matrix multiply(double v) {
        return iterator(multiplyValues, v);
    }

    public final Matrix divide(double v) {
        return iterator(divideValues, v);
    }

    public final Matrix square() {
        return iterator(squareValue);
    }

    public final Matrix apply(ApplyValueFunction function) {
        return iterator(function);
    }

    public final Matrix copy() { 
        return new Matrix(this.arrayData); 
    }

    public final double[][] get() {
        return this.arrayData;
    }

    public final double[] get(int row) {
        return this.arrayData[row];
    }

    public final double get(int row, int col) {
        return this.arrayData[row][col];
    }

    public final void set(int row, double[] v) {
        this.arrayData[row] = v;
    }

    public final void set(int row, int col, double v) {
        this.arrayData[row][col] = v;
    }

    public static final Matrix subtract(Matrix a, Matrix b) {
        return new Matrix(a).iterator(subtractValues, b);
    }

    public static final Matrix add(Matrix a, Matrix b) {
        return new Matrix(a).iterator(addValues, b);
    }

    public static final Matrix apply(Matrix matrix, ApplyValueFunction function) {
        return new Matrix(matrix).iterator(function);
    }

    /**
     * <p>
     * Applies output from provided function to 
     * each matrix entry and provided value.
     * </p>
     * 
     * <p>
     * <b>Note:</b> Assumes that function takes provided value as input.
     * </p>
     * 
     * @param matrix        target matrix
     * @param value         value to be provided when calling {@param function}
     * @param function      function to be applied on {@param matrix} with provided {@param value}
     * @return              mutated instance of Matrix object
     * @see                 #apply(Matrix, ApplyValueFunction)
     */
    public static final Matrix apply(Matrix matrix, double value, ApplyValuesFunction function) {
        return new Matrix(matrix).iterator(function, value);
    }

    public static final Matrix multiply(Matrix a, Matrix b) {
        Matrix res = new Matrix(a.rows, b.cols);
        for (int row = 0; row < res.rows; row++) {  
            for (int col = 0; col < res.cols; col++) {
                double v = 0.0;
                for (int i = 0; i < a.cols; i++) v += a.get(row, i) * b.get(i, col);
                res.arrayData[row][col] = v;
            }
        }
        return res;
    }

    public static final Matrix dot(Matrix a, Matrix b) {
        Matrix res = new Matrix(a.rows, a.cols);
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.arrayData[i][j] = a.get(i, j) * b.get(i, j);
            }
        }
        return res;
    }

    public static final Matrix divide(Matrix matrixA, Matrix matrixB) {
        
        Matrix res = new Matrix(matrixA.rows, matrixA.cols);

        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.arrayData[i][j] = matrixA.arrayData[i][j] / matrixB.arrayData[i][j];
            }
        }

        return res;

    }

    /**
     * Calculate sum of matrix over specified axis.
     * 
     * @param matrix    target matrix
     * @param axis      target axis
     * @return          sum of {@param matrix} over provided {@param axis}
     * @see             #sum(Matrix)
     */
    public static final Matrix sum(Matrix matrix, int axis) {
        
        assert((axis == 0) || (axis == 1));

        int rows = axis == 0 ? 1 : matrix.rows;
        int cols = axis == 1 ? 1 : matrix.cols;

        Matrix res = new Matrix(rows, cols);

        for (int i = 0; i < (axis == 0 ? cols : rows); i++) {  
            double sum = 0.0;
            for (int j = 0; j < (axis == 0 ? matrix.rows : matrix.cols); j++) {  
                sum += (axis == 0 ? matrix.arrayData[j][i] : matrix.arrayData[i][j]);
            }
            res.arrayData[axis == 0 ? 0 : i][axis == 0 ? i : 0] = sum;
        }

        return res;

    }

    /**
     * Calculate sum of matrix.
     * 
     * @param matrix    target matrix
     * @return          sum of {@param matrix}
     * @see             #sum(Matrix, Integer)
     */
    public static final double sum(Matrix matrix) {
        double sum = 0.0;
        for (int i = 0; i < matrix.rows; i++) {  
            for (int j = 0; j < matrix.cols; j++) {  
                sum += matrix.arrayData[i][j];
            }
        }
        return sum;
    }

    public static final Matrix mean(Matrix matrix, int axis) {
        
        assert((axis == 0) || (axis == 1));

        final int n = axis == 0 ? matrix.rows : matrix.cols;

        int rows = axis == 0 ? 1 : matrix.rows;
        int cols = axis == 1 ? 1 : matrix.cols;

        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < (axis == 0 ? cols : rows); i++) {  
            double sum = 0.0;
            for (int j = 0; j < (axis == 0 ? matrix.rows : matrix.cols); j++) {  
                sum += (axis == 0 ? matrix.arrayData[j][i] : matrix.arrayData[i][j]);
            }
            res.arrayData[axis == 0 ? 0 : i][axis == 0 ? i : 0] = sum / n;
        }

        return res;

    }

    public static final Matrix transpose(Matrix matrix) {

        Matrix res = new Matrix(matrix.cols, matrix.rows);

        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.arrayData[i][j] = matrix.arrayData[j][i];
            }
        }

        return res;

    }

    public static final Matrix repeat(Matrix matrix, int axis, int n) {

        Matrix res = new Matrix(axis == 0 ? n : matrix.rows, axis == 1 ? n : matrix.cols);

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < (axis == 0 ? matrix.cols : matrix.rows); i++) {
                res.arrayData[axis == 0 ? k : i][axis == 0 ? i : k] = matrix.arrayData[axis == 0 ? 0 : i][axis == 0 ? i : 0];
            }
        }

        return res;

    }

    public static final Matrix getIndices(Matrix matrix, int axis, Integer[] indices) {

        int n = indices.length;

        Matrix res = new Matrix(axis == 0 ? n : matrix.rows, axis == 1 ? n : matrix.cols);

        for (int i = 0; i < (axis == 0 ? n : matrix.rows); i++) {
            for (int j = 0; j < (axis == 1 ? n : matrix.cols); j++) {
                res.arrayData[i][j] = matrix.get(axis == 0 ? indices[i] : i, axis == 0 ? j : indices[j]);
            }
        }

        return res;

    }

    public static final Matrix shuffle(Matrix matrix, int axis, int n) {

        assert(((axis == 0) && (n < matrix.rows)) || ((axis == 1) && (n < matrix.cols)));

        return getIndices(matrix, axis, MatrixData.getRandomIndices(n));

    }

    public static final Matrix replace(Matrix source, double value, int[][] indices) {
        
        Matrix res = new Matrix(source);

        for (int i = 0; i < indices.length; i++) {
            res.arrayData[indices[i][0]][indices[i][1]] = value;
        }
        
        return res;

    }

    public static final Matrix replace(Matrix source, double value, int n) {
        return replace(source, value, MatrixData.getRandomMatrixIndices(source, n));
    }

    public static final Matrix replace(Matrix source, Matrix target, int axis, Integer[] indices) {

        assert((source.rows == target.rows) || (source.cols == target.cols));
        
        Matrix res = new Matrix(source);

        if (axis == 0) {
            for (int i = 0; i < indices.length; i++) {  
                for (int j = 0; j < res.cols; j++) {  
                    res.arrayData[indices[i]][j] = target.arrayData[indices[i]][j];
                }
            }
        } else {
            for (int i = 0; i < res.rows; i++) {  
                for (int j = 0; j < indices.length; j++) {  
                    res.arrayData[i][indices[j]] = target.arrayData[i][indices[j]];
                }
            }
        }
        
        return res;

    }

    public static final Matrix slice(Matrix matrix, int axis, int fromIndex, int toIndex) {

        assert(toIndex > fromIndex);
        assert((axis == 0) || (axis == 1));
        assert(!((axis == 0) && (fromIndex < 0) && (toIndex >= matrix.rows)));
        assert(!((axis == 1) && (fromIndex < 0) && (toIndex >= matrix.cols)));

        int n = toIndex - fromIndex;

        Matrix res = new Matrix(axis == 0 ? n : matrix.rows, axis == 1 ? n : matrix.cols);

        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.arrayData[i][j] = matrix.arrayData[i][j];
            }
        }

        return res;

    }

    public static final Matrix diag(double value, int n) {

        Matrix res = new Matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res.arrayData[i][j] = i == j ? value : 0;
            }
        }

        return res;

    }

    public static final Matrix fromArray(double[][] arrayData) {
        return new Matrix(arrayData);
    }

    public static final double[][] copyArray(double[][] arrayData) {
        final double[][] arrayDataCopy = new double[arrayData.length][arrayData[0].length];
        for (int i = 0; i < arrayData.length; i++) {
            for (int j = 0; j < arrayData[0].length; j++) {
                arrayDataCopy[i][j] = arrayData[i][j];
            }
        }
        return arrayDataCopy;
    }

    @Override
    public String toString() {
        return Ascii.MatrixAscii.getString(this, this.label == null ? "Matrix" : this.label);
    }

    public String toString(String header, int r) {
        return Ascii.MatrixAscii.getString(this, header, r);
    }

    public String toString(int r) {
        return Ascii.MatrixAscii.getString(this, this.label == null ? "Matrix" : this.label, r);
    }

    public String toString(String header) {
        return Ascii.MatrixAscii.getString(this, header);
    }

    public void print(String header, int r) {
        System.out.println(toString(header, r));
    }

    public void print(String header) {
        System.out.println(toString(header));
    }

    public void print(int r) {
        print(this.label == null ? "Matrix" : this.label, r);
    }

    public void print() {
        print(this.label == null ? "Matrix" : this.label);
    }


    
    public static final void main(String[] args) {

        Matrix a = Matrix.fromArray(new double[][] {{3, 2}, {1, 2}});
        Matrix b = Matrix.fromArray(new double[][] {{2, 1}, {1, 1}});

        a.label = "A";
        b.label = "B";

        a.print();
        b.print();

        divide(a, b).print("A / B");

        a.add(12).print("A = A + 12");
        a.divide(10).print("A = A / 10");
        a.multiply(10).print("A = A * 10");
        a.subtract(12).print("A = A - 12");
        a.square().print("A_ij = A_ij * A_ij");
        a.print("A");

        multiply(a, b).print("A = A * B");
        a.print("A");
        transpose(a).print("A = A^T");
        a.print("A");
        dot(a, b).print("A . T");
        a.print("A");
        
        Matrix c = Matrix.fromArray(new double[][] {{2, 1, 1}});
        Matrix d = Matrix.fromArray(new double[][] {{4, 4}});

        c.print("c");
        d.print("d");

        multiply(transpose(c), d).print("C^T * D");;

        a.print("A");
        sum(a, 0).print("A.sum(axis = 0)");
        sum(a, 1).print("A.sum(axis = 1)");

        a.print("A");
        System.out.println(sum(a));

        repeat(a, 0, 4).print(null);
        repeat(a, 1, 2).print(null);

        sum(b, 1).print("sum(B, 1)");
        mean(b, 1).print("mean(B, 1)");

        b = sum(b, 1);
        b.print("B = B.sum(1)");
        repeat(b, 1, 2).print(null);;

        Matrix e = Matrix.fromArray(new double[][] {{2, 1, 4}, {1, 1, 23}, {-1, 2, 9}, {-1, -2, 3}, {4, 8, 9}});
        Matrix f = Matrix.fromArray(new double[][] {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
        e.print("E");
        shuffle(e, 0, 3).print("shuffle(E, 0, 3)");
        shuffle(e, 1, 2).print("shuffle(E, 1, 2)");

        e.print("E");
        f.print("F");
        replace(e, f, 0, new Integer[] {0, 1, 4}).print("replace(E, F, 0, ...)");
        replace(e, f, 1, new Integer[] {1}).print("replace(E, F, 1, ...)");

        e.print("E");
        slice(e, 0, 4, 5).print("slice(E, 0, 4, 5)");
        slice(e, 1, 0, 3).print("slice(E, 1, 0, 3)");

        e.print("E");
        int[][] randomIndices = MatrixData.getRandomMatrixIndices(e, 4);
        for (int i = 0; i < randomIndices.length; i++) {
            for (int j = 0; j < randomIndices[0].length; j++) {
                System.out.print(randomIndices[i][j] + "   ");
            }
            System.out.println();
        }
        replace(e, 0, randomIndices).print("replace(E, 0, randomIndices)");
        replace(e, 42, 4).print("replace(E, 42, 4)");

        diag(1, 5).print("I(5)", 0);

    }

}
