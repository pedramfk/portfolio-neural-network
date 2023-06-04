package se.pedramfk.portfolio.ml.ann.results;

import java.util.HashMap;
import java.util.Map;
import se.pedramfk.portfolio.ml.utils.Matrix;


public class LayerResults {

    private final Map<Integer, Matrix> weigths;
    private final Map<Integer, Matrix> biases;

    public LayerResults() {
        this.weigths = new HashMap<>();
        this.biases = new HashMap<>();
    }

    public Matrix getWeight(int epoch) {
        return this.weigths.get(epoch);
    }

    public Matrix getBias(int epoch) {
        return this.biases.get(epoch);
    }
    
}
