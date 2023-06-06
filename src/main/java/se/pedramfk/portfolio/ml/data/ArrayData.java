package se.pedramfk.portfolio.ml.data;

public interface ArrayData {

    @FunctionalInterface public static interface ApplyVoidFunction { double apply(); }
    @FunctionalInterface public static interface ApplyValueFunction { double apply(double value); }
    @FunctionalInterface public static interface ApplyValuesFunction { double apply(double value1, double value2); }

    public ArrayData add(double v);

    public ArrayData add(ArrayData d);

    public ArrayData subtract(double v);

    public ArrayData subtract(ArrayData d);

    public ArrayData multiply(double v);

    public ArrayData multiply(ArrayData d);
    
}
