package se.pedramfk.portfolio.ann.layers;

import se.pedramfk.portfolio.ann.utils.MatrixData;


/**
 * 
 * X: [numberOfSamples, inputDim]
 * W: [numberOfNodes, inputDim]
 * b: [numberOfNodes, 1]
 * y = W * X^T + b
 * 
 */
public interface Layer {

    //public MatrixData getInputData();

    //public MatrixData getOutputData();

    public MatrixData forwardPropagate(MatrixData x);

    public MatrixData backwardPropagate(MatrixData target, MatrixData output, double lr);

    public MatrixData backwardPropagate(MatrixData error, double lr);
    

}
