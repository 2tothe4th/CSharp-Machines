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
        public double[] CalculateWeightedSums(double[] inputs)
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
            }
            return outputs;
        }
        public static double[] CalculateOutputsFromWeightedSums(double[] sums, Func<double, double> activationFunction)
        {
            double[] outputs = new double[sums.Length];
            for (int i = 0; i < sums.Length; i++)
            {
                outputs[i] = activationFunction(sums[i]);
            }
            return outputs;
        }
        public double[] CalculateOutputsFromInputs(double[] inputs, Func<double, double> activationFunction)
        {
            return CalculateOutputsFromWeightedSums(CalculateWeightedSums(inputs), activationFunction);
        }
    }
}
