package se.pedramfk.portfolio.ml.utils;


public interface MatrixFunctions extends ApplyFunctions {
    
    static final ApplyVoidFunction initWithRandom = () -> Math.random() - .5;

    static final ApplyValueFunction initWithRandomScaling = (scale) -> scale * (Math.random() - .5);
    static final ApplyValueFunction initWithValue = (value) -> value;
    static final ApplyValueFunction squareValue = (value) -> value * value;

    static final ApplyValuesFunction addValues = (v1, v2) -> v1 + v2;
    static final ApplyValuesFunction subtractValues = (v1, v2) -> v1 - v2;
    static final ApplyValuesFunction multiplyValues = (v1, v2) -> v1 * v2;
    static final ApplyValuesFunction divideValues = (v1, v2) -> v1 / v2;

}
