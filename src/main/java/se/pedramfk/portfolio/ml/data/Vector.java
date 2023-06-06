package se.pedramfk.portfolio.ml.data;


public class Vector implements ArrayData {

    public final int size;
    public double[] arrayData;
    public String label;

    public Vector(double[] arrayData, String label) {
        this.size = arrayData.length;
        this.arrayData = arrayData;
        this.label = label;
    }

    private static final ApplyValuesFunction addValues = (v1, v2) -> v1 + v2;
    private static final ApplyValuesFunction subtractValues = (v1, v2) -> v1 - v2;
    private static final ApplyValuesFunction multiplyValues = (v1, v2) -> v1 * v2;

    public double get(int i) {
        return this.arrayData[i];
    }

    public static final double[] copyArray(double[] arrayData) {
        double[] arrayDataCopy = new double[arrayData.length];
        for (int i = 0; i < arrayData.length; i++) {
            arrayDataCopy[i] = arrayData[i];
        }
        return arrayDataCopy;
    }

    public final Vector iterator(ApplyVoidFunction f) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply();
        }
        return this;        
    }

    public final Vector iterator(ApplyValueFunction f) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply(this.arrayData[i]);
        }
        return this;  
    }

    public final Vector iterator(ApplyValueFunction f, double v) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply(v);
        }
        return this;  
    }

    public final Vector iterator(ApplyValuesFunction f, double v) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply(this.arrayData[i], v);
        }
        return this;
    }

    public final Vector iterator(ApplyValuesFunction f, Vector d) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply(this.arrayData[i], d.get(i));
        }
        return this;  
    }

    public final Vector iterator(ApplyValuesFunction f, double v1, double v2) {
        for (int i = 0; i < arrayData.length; i++) {
            this.arrayData[i] = f.apply(v1, v2);
        }
        return this;
    }

    @Override public ArrayData add(double v) { return iterator(addValues, v); }
    @Override public ArrayData subtract(double v) { return iterator(subtractValues, v); }
    @Override public ArrayData multiply(double v) { return iterator(multiplyValues, v); }

    @Override
    public ArrayData add(ArrayData d) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'add'");
    }

    @Override
    public ArrayData subtract(ArrayData d) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'subtract'");
    }

    @Override
    public ArrayData multiply(ArrayData d) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'multiply'");
    }
   
}
