package se.pedramfk.portfolio.ann.utils;

import java.util.function.Function;

public final class ActivationFunctions {

    public static final Function<Double, Double> SIGMOID = v -> 1.0 / (1.0 + Math.exp(-v));

    //public static final Function<Double, Double> SIGMOID_D = v -> (1.0 / (1.0 + Math.exp(-v))) * (1.0 - (1.0 / (1.0 + Math.exp(-v))));
    public static final Function<Double, Double> SIGMOID_D = v -> v * (1.0 - v);

    public static final Function<Double, Double> TANH = v -> Math.tanh(v);

    public static final Function<Double, Double> TANH_D = v -> 1.0 - Math.pow(Math.tanh(v), 2);
    
}
