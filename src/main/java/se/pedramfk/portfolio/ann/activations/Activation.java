package se.pedramfk.portfolio.ann.activations;

import java.util.function.Function;

public final class Activation {

    public static final Function<Double, Double> SIGMOID = v -> 1.0 / (1.0 + Math.exp(-v));
    public static final Function<Double, Double> DELTA_SIGMOID = v -> v * (1.0 - v);

    public static final Function<Double, Double> TANH = v -> Math.tanh(v);
    public static final Function<Double, Double> DELTA_TANH = v -> 1.0 - Math.pow(Math.tanh(v), 2);
    
}
