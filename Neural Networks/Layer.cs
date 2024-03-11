using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    public class Layer
    {
        public double[,] weights;
        public double[] biases;
        public double[] CalculateOutputs(double[] inputs, Func<double, double> activation)
        {
            double[] outputs = new double[weights.GetLength(1)];
            for (int i = 0; i < inputs.Length; i++)
            {
                for (int j = 0; j < outputs.Length; j++)
                {
                    outputs[j] += inputs[i] * weights[i, j];
                }
            }
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] += biases[i];
                outputs[i] = activation(outputs[i]);
            }
            return outputs;
        }
    }
}
