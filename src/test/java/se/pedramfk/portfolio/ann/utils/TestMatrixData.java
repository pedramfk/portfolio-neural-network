package se.pedramfk.portfolio.ann.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestMatrixData {

    @Test
    public void testDimensions() {

        final double[] a = {2.3, 9, 8, 4};
        final double[][] b = {a};

        final MatrixData matrixA = new MatrixData(a);
        final MatrixData matrixB = new MatrixData(b);

        assertEquals(a.length, matrixA.nCols);
        assertEquals(1, matrixA.nRows);

        assertEquals(b[0].length, matrixB.nCols);
        assertEquals(1, matrixB.nRows);

    }

    @Test
    public void testAdd() {

        final double[][] a = {
            {3, 9, 8, 4}, 
            {2, 8, 7, 3}, 
            {1, 7, 6, 2}
        };

        final double[][] b = {
            {2, 8, 7, 3}, 
            {1, 7, 6, 2}, 
            {0, 6, 5, 1}
        };

        final MatrixData matrixA = new MatrixData(a);
        final MatrixData matrixB = new MatrixData(b);
        final MatrixData matrixC = MatrixData.add(matrixA, matrixB);

        for (int r = 0; r < matrixC.nRows; r++) {
            for (int c = 0; c < matrixC.nCols; c++) {
                assertEquals(a[r][c] + b[r][c], matrixC.get(r, c));
            }
        }

        matrixA.add(matrixB);
        for (int r = 0; r < matrixC.nRows; r++) {
            for (int c = 0; c < matrixC.nCols; c++) {
                assertEquals(a[r][c] + b[r][c], matrixA.get(r, c));
            }
        }

    }

}
