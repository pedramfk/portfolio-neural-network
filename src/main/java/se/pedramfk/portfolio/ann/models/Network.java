package se.pedramfk.portfolio.ann.models;

import java.util.List;
import java.util.ArrayList;

import se.pedramfk.portfolio.ann.activations.Activation;
import se.pedramfk.portfolio.ann.layers.Layer;
import se.pedramfk.portfolio.ann.layers.dense.DenseLayer;
import se.pedramfk.portfolio.ann.utils.MatrixData;
import se.pedramfk.portfolio.ann.utils.LossFunctions;


public class Network {

    private final List<Layer> layers = new ArrayList<>();

    public Network() {

    }

    public final void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    private final MatrixData getForwardProp(MatrixData x) {

        MatrixData output = new MatrixData(x);
        for (Layer layer: layers) {
            output = new MatrixData(layer.forwardPropagate(output));
        }
        return output;

    }

    private final MatrixData getBackwardProp(MatrixData error, double lr) {

        //MatrixData deltaMse = new MatrixData(LossFunctions.getDeltaMSE(y[i], yPred[i]));

        //MatrixData output = new MatrixData(yDelta);
        
        for (int i = layers.size() - 1; i >= 0 ; i--) {
            error = layers.get(i).backwardPropagate(error, lr);
        }

        return error;

    }

    public final MatrixData predict(double[][] inputData) {

        double[][] yPred = new double[inputData.length][1];
        for (int i = 0; i < yPred.length; i++) {
            yPred[i] = getForwardProp(new MatrixData(inputData[i])).get(0);
        }
        return new MatrixData(yPred);

    }

    public final void fit(double[][] x, double[][] y, int epochs, double lr) {

        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {

            MatrixData yMSE = new MatrixData(y.length, y[0].length).initializeWithValue(0);
            for (int i = 0; i < x.length; i++) {
                
                MatrixData yPred = getForwardProp(new MatrixData(x[i]));
                MatrixData yActual = new MatrixData(y[i]);
                double mse = LossFunctions.getMSE(yActual, yPred);
                yMSE.set(i, 0, mse);
                
                //yPred[i] = output.get(0);
                //mse += LossFunctions.getMSE(y[i], yPred.get(i));
                
                // Backward Propagate
                getBackwardProp(MatrixData.subtract(yActual, yPred), lr);
                //getBackwardProp(LossFunctions.getDeltaMSE(yActual, yPred), lr);
                
            }

            //yMSE = yMSE.divide(x.length);

            System.out.println(String.format("epoch = %d\terror = %f", currentEpoch, yMSE.get(0, 0)));

        }


    }

    public static final void main(String[] args) {

        double[][] x = {
            {0, 0}, 
            {0, 1}, 
            {1, 0}, 
            {1, 1}
        };

        double[][] y = {
            {0}, 
            {1}, 
            {1}, 
            {0}
        };

        Network network = new Network() {{
            addLayer(new DenseLayer(2, 3, Activation.TANH, Activation.DELTA_TANH));
            //addLayer(new DenseLayer(3, 1, Activation.TANH, Activation.DELTA_TANH));
            addLayer(new DenseLayer(3, 1, Activation.SIGMOID, Activation.DELTA_SIGMOID));
            //addLayer(new FullyConnectedLayer(2, 3));
            //addLayer(new SigmoidLayer());
            //addLayer(new FullyConnectedLayer(3, 1));
            //addLayer(new SigmoidLayer());
        }};

        network.predict(x).print("Prediction");

        network.fit(x, y, 100000, 1e-4);

        network.predict(x).print("Prediction");

    }

}
/*
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)



    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
 */