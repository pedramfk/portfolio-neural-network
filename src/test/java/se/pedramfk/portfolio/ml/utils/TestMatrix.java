package se.pedramfk.portfolio.ml.utils;

import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static se.pedramfk.portfolio.ml.utils.Matrix.add;
import static se.pedramfk.portfolio.ml.utils.Matrix.dot;
import static se.pedramfk.portfolio.ml.utils.Matrix.subtract;
import static se.pedramfk.portfolio.ml.utils.Matrix.multiply;
import static se.pedramfk.portfolio.ml.utils.Matrix.transpose;



@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestMatrix {
    
    private static final double[][] ARRAY_1_4 = {{2.3, 9, 8, 4}};
    private static final double[][] ARRAY_4_1 = {{0}, {1}, {1}, {2}};
    private static final double[][] ARRAY_4_1_T = {{0, 1, 1, 2}};
    private static final double[][] ARRAY_1_4_4_1_MULT = {{25}};

    private static final double[][] ARRAY_3_3 = {
        {3, 1, 8}, 
        {2, 1, 0}, 
        {1, 0, 6}
    };

    private static final double[][] ARRAY_3_I = {
        {1, 0, 0}, 
        {0, 1, 0}, 
        {0, 0, 1}
    };

    private static final double[][] ARRAY_3_2 = {
        {2, 0}, 
        {1, 1}, 
        {0, 2}
    };

    private static final double[][] ARRAY_3_I_3_3_MULT = {
        {3, 1, 8}, 
        {2, 1, 0}, 
        {1, 0, 6}
    };

    private static final double[][] ARRAY_3_3_3_2_MULT = {
        {7, 17}, 
        {5, 1}, 
        {2, 12}
    };

    private static final Stream<Arguments> matrices() {
        return Stream.of(
            Arguments.of(ARRAY_1_4, ARRAY_4_1, ARRAY_3_3, ARRAY_3_I, ARRAY_3_2)
        );
    }

    private static final Stream<Arguments> matrixMultiplications() {
        return Stream.of(
            Arguments.of(ARRAY_1_4, ARRAY_4_1, ARRAY_1_4_4_1_MULT), 
            Arguments.of(ARRAY_3_I, ARRAY_3_3, ARRAY_3_I_3_3_MULT), 
            Arguments.of(ARRAY_3_3, ARRAY_3_2, ARRAY_3_3_3_2_MULT)
        );
    }

    private static final Stream<Arguments> matrixProducts() {
        return Stream.of(
            Arguments.of(ARRAY_1_4, ARRAY_4_1_T), 
            Arguments.of(ARRAY_3_3, ARRAY_3_I)
        );
    }

    private static final Stream<Arguments> matrixAdditions() {
        return Stream.of(
            Arguments.of(ARRAY_1_4, ARRAY_1_4), 
            Arguments.of(ARRAY_3_3, ARRAY_3_I)
        );
    }

    private static final Stream<Arguments> matrixSubtractions() {
        return Stream.of(
            Arguments.of(ARRAY_1_4, ARRAY_1_4), 
            Arguments.of(ARRAY_3_3, ARRAY_3_I)
        );
    }

    @DisplayName("Checking dimensions")
    @ParameterizedTest
    @MethodSource("matrices")
    public void testDimensions(double[][] arrayData) {

        Matrix matrixData = new Matrix(arrayData);

        assertEquals(arrayData.length, matrixData.rows);
        assertEquals(arrayData[0].length, matrixData.cols);

        Matrix transposedMatrixData = transpose(matrixData);

        assertEquals(arrayData[0].length, transposedMatrixData.rows);
        assertEquals(arrayData.length, transposedMatrixData.cols);

    }

    @DisplayName("Checking multiplication")
    @ParameterizedTest
    @MethodSource("matrixMultiplications")
    public void testMultiply(double[][] arrayDataA, double[][] arrayDataB, double[][] expected) {

        Matrix res = multiply(new Matrix(arrayDataA), new Matrix(arrayDataB));

        for (int r = 0; r < res.rows; r++) {
            for (int c = 0; c < res.cols; c++) {
                assertEquals(res.get(r, c), expected[r][c]);
            }
        }

    }

    @DisplayName("Checking dot product")
    @ParameterizedTest
    @MethodSource("matrixProducts")
    public void testProduct(double[][] arrayDataA, double[][] arrayDataB) {

        Matrix res = dot(new Matrix(arrayDataA), new Matrix(arrayDataB));

        for (int r = 0; r < res.rows; r++) {
            for (int c = 0; c < res.cols; c++) {
                assertEquals(res.get(r, c), arrayDataA[r][c] * arrayDataB[r][c]);
            }
        }

    }

    @DisplayName("Checking addition")
    @ParameterizedTest
    @MethodSource("matrixAdditions")
    public void testAdd(double[][] arrayDataA, double[][] arrayDataB) {

        Matrix res = add(new Matrix(arrayDataA), new Matrix(arrayDataB));

        for (int r = 0; r < res.rows; r++) {
            for (int c = 0; c < res.cols; c++) {
                assertEquals(res.get(r, c), arrayDataA[r][c] + arrayDataB[r][c]);
            }
        }

    }

    @DisplayName("Checking subtraction")
    @ParameterizedTest
    @MethodSource("matrixSubtractions")
    public void testSubtract(double[][] arrayDataA, double[][] arrayDataB) {

        Matrix res = subtract(new Matrix(arrayDataA), new Matrix(arrayDataB));

        for (int r = 0; r < res.rows; r++) {
            for (int c = 0; c < res.cols; c++) {
                assertEquals(res.get(r, c), arrayDataA[r][c] - arrayDataB[r][c]);
            }
        }

    }
    
}
