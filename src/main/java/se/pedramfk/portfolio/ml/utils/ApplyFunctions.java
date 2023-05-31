package se.pedramfk.portfolio.ml.utils;


public interface ApplyFunctions {

    @FunctionalInterface
    public static interface ApplyVoidFunction { 
        double apply(); 
    }

    @FunctionalInterface
    public static interface ApplyValueFunction { 
        double apply(double value); 
    }

    @FunctionalInterface
    public static interface ApplyValuesFunction { 
        double apply(double value1, double value2);
    }
    
}
