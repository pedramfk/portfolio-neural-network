package se.pedramfk.portfolio.ml.ann.activations;

import se.pedramfk.portfolio.ml.utils.Matrix;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValueFunction;
import se.pedramfk.portfolio.ml.utils.ApplyFunctions.ApplyValuesFunction;

public final class SoftMaxActivation implements Activation {

    private static final ApplyValueFunction exp = (v) -> Math.exp(v);
    private static final ApplyValuesFunction softmax = (v, vSum) -> v / vSum;
    //private static final ApplyValuesFunction softmaxGradient = (v) -> v / vSum;

    @Override
    public Matrix getActivation(Matrix z) {
        Matrix zExp = Matrix.apply(z, exp);
        return Matrix.apply(zExp, Matrix.sum(zExp), softmax);
    }

    /**
def softmax_grad(s):
    # input s is softmax value of the original input x. Its shape is (1,n) 
    # i.e.  s = np.array([0.3,0.7]),  x = np.array([0,1])

    # make the matrix whose size is n^2.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else: 
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m
     */

    @Override
    public Matrix getActivationGradient(Matrix z) {

        Matrix zMean = z.cols > 1 ? Matrix.mean(z, 1) : z;
        
        Matrix gradient = new Matrix(zMean.rows, zMean.rows);

        //Matrix g = Matrix.diag(1, z.rows);

        for (int i = 0; i < gradient.rows; i++) {
            for (int j = 0; j < gradient.cols; j++) {
                double vi = zMean.get(i, 0);
                if (i == j) {
                    gradient.set(j, i, vi * (1 - vi));
                } else {
                    double vj = zMean.get(j, 0);
                    gradient.set(j, i, - vi * vj);
                }
            }
        }

        return gradient;

        //return Matrix.subtract(Matrix.diag(1, z.rows), Matrix.multiply(z, Matrix.transpose(z)));

    }
    
    public static final void main(String[] args) {

        Matrix matrix = new Matrix(new double[][] {
            {1, 2}, {2, 1}, 
            //{2, 3, 1}
        });

        matrix.label = "Z";
        
        SoftMaxActivation softMaxActivation = new SoftMaxActivation();
        
        Matrix z = softMaxActivation.getActivation(matrix);
        z.print("softmax(Z)", 3);

        Matrix zGrad = softMaxActivation.getActivationGradient(z);
        zGrad.print("softmax'(Z)", 3);
    }

}
