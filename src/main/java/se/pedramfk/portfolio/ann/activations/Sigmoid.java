package se.pedramfk.portfolio.ann.activations;


public final class Sigmoid {

    public static final double get(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    public static final double getDelta(double v) {
        return v - (1.0 - v);
    }
    
}
