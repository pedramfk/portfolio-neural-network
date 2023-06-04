package se.pedramfk.portfolio.ml.ann.optimizers;

import se.pedramfk.portfolio.ml.ann.layers.Layer;
import se.pedramfk.portfolio.ml.utils.Matrix;


public interface Optimizer {

    public double getCurrentLearningRate();

    public boolean decayLearningRate();

    public double getMomentum();

    public boolean useMomentum();

    public Matrix step(int iteration, Layer layer, Matrix gradient);
    
}
