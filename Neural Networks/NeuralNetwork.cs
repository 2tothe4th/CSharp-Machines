using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    public class NeuralNetwork
    {
        public Layer[] layers;
        public int[] nodeCounts;
        public double learningStrength = 0.5;
        public double weightAndBiasRandomRange = 1;

        //https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch
        public double miniBatchSize = 32;

        //Partial derivative calculations
        public double deltaW = 0.01;
        public double deltaB = 0.01;

        //Activation functions
        Func<double, double> activationFunction = input => ActivationFunctions.Sigmoid(input);

        //public double[] inputValues;
        public void CreateLayers()
        {
            int numberOfLayers = nodeCounts.Length;
            layers = new Layer[numberOfLayers];
            for (int i = 0; i < numberOfLayers - 1; i++)
            {
                layers[i] = new Layer();
                layers[i].weights = new double[nodeCounts[i], nodeCounts[i + 1]];
                layers[i].biases = new double[nodeCounts[i + 1]];
            }

            layers[numberOfLayers - 1] = new Layer();
            layers[nodeCounts.Length - 1].weights = new double[nodeCounts[numberOfLayers - 1], 1];
            layers[nodeCounts.Length - 1].biases = new double[1];
            for (int i = 0; i < nodeCounts[numberOfLayers - 1]; i++)
            {
                layers[numberOfLayers - 1].weights[i, 0] = 1;
                //layers[numberOfLayers - 1].biases[i] = 1;
            }
        }
        public double[] CalculateOutputsForLayer(double[] inputValues, int layer)
        {
            if (layer == 0)
                return layers[0].CalculateOutputs(inputValues, activationFunction);
            return layers[layer].CalculateOutputs(CalculateOutputsForLayer(inputValues, layer - 1), activationFunction);
        }
        public double CalculateCostForDataPoint(DataPoint data)
        {
            double[] calculatedOutputs = CalculateOutputsForLayer(data.inputs, nodeCounts.Length - 1);
            double cost = 0;
            for (int i = 0; i < data.outputs.Length; i++)
            {
                double error = calculatedOutputs[i] - data.outputs[i];
                cost += error * error;
            }
            return cost;
        }
        public double CalculateCostForDataSet(DataPoint[] dataSet)
        {
            var costs = new List<double>();
            foreach (DataPoint currentDataPoint in dataSet)
            {
                costs.Add(CalculateCostForDataPoint(currentDataPoint));
            }
            return costs.Average();
        }
        public void RandomizeWeightsAndBiases()
        {
            //https://stackoverflow.com/questions/8308834/creating-a-true-random
            Random random = new Random(DateTime.Now.Millisecond);
            for (int layer = 0; layer < nodeCounts.Length - 1; layer++)
            {
                Layer currentLayer = layers[layer];
                for (int outNode = 0; outNode < currentLayer.weights.GetLength(1); outNode++)
                {
                    currentLayer.biases[outNode] = random.NextDouble() * weightAndBiasRandomRange * 2 - weightAndBiasRandomRange;
                    for (int inNode = 0; inNode < currentLayer.weights.GetLength(0); inNode++)
                    {
                        currentLayer.weights[inNode, outNode] = random.NextDouble() * weightAndBiasRandomRange * 2 - weightAndBiasRandomRange;
                    }
                }
            }
        }
        public double CalculateActivationOverBiasDerivative(int layer, int inNode, int outNode, double previousActivation)
        {
            //z = wx + b
            //a = A(z)
            Layer currentLayer = layers[layer];
            return ActivationFunctionDerivatives.Sigmoid(currentLayer.weights[inNode, outNode] * previousActivation + currentLayer.biases[outNode]);
        }
        public double CalculateActivationOverWeightDerivative(int layer, int inNode, int outNode, double previousActivation)
        {
            //z = wx + b
            //a = A(z)
            Layer currentLayer = layers[layer];
            return CalculateActivationOverBiasDerivative(layer, inNode, outNode, previousActivation)
                * previousActivation;
        }
        public double CalculateNewActivationOverLastActivationDerivative(int layer, int inNode, int outNode, double previousActivation)
        {
            //z = wx + b
            //a = A(z)
            Layer currentLayer = layers[layer];
            return CalculateActivationOverBiasDerivative(layer, inNode, outNode, previousActivation)
                * currentLayer.weights[inNode, outNode];
        }
        public double CalculateDataPointCostOverActivationDerivative(double activation, double[] expectedOutputs)
        {
            double halfOfPartialDerivative = 0;
            //https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3?gi=0b203f4b1ff9#:~:text=The%20Mean%20Squared%20Error%20(MSE,out%20across%20the%20whole%20dataset.
            foreach (double currentExpectedOutput in expectedOutputs)
            {
                halfOfPartialDerivative += activation - currentExpectedOutput;
            }
            return 2 * halfOfPartialDerivative;
        }
        public double CalculateDataSetCostOverBiasDerivative(DataPoint[] dataSet, int layer, int inNode, int outNode)
        {
            List<double> costDerivatives = new List<double>();
            foreach (DataPoint currentDataPoint in dataSet)
            {
                double previousActivation = CalculateOutputsForLayer(currentDataPoint.inputs, layer)[outNode];
                double partialDerivative = CalculateDataPointCostOverActivationDerivative(previousActivation, currentDataPoint.outputs)
                    * CalculateActivationOverBiasDerivative(layer, inNode, outNode, previousActivation);
                for (int currentLayerNumber = layer + 1; currentLayerNumber < nodeCounts.Length - 2; currentLayerNumber++)
                {
                    partialDerivative *= CalculateNewActivationOverLastActivationDerivative(currentLayerNumber, inNode, outNode, previousActivation);
                }
                costDerivatives.Add(partialDerivative);
            }
            return costDerivatives.Average();
        }
        public double CalculateDataSetCostOverWeightDerivative(DataPoint[] dataSet, int layer, int inNode, int outNode)
        {
            List<double> costDerivatives = new List<double>();
            foreach (DataPoint currentDataPoint in dataSet)
            {
                double previousActivation = CalculateOutputsForLayer(currentDataPoint.inputs, layer)[outNode];
                double partialDerivative = CalculateDataPointCostOverActivationDerivative(previousActivation, currentDataPoint.outputs)
                    * CalculateActivationOverWeightDerivative(layer, inNode, outNode, previousActivation)
                    * previousActivation;
                for (int currentLayerNumber = layer + 1; currentLayerNumber < nodeCounts.Length - 2; currentLayerNumber++)
                {
                    partialDerivative *= CalculateNewActivationOverLastActivationDerivative(currentLayerNumber, inNode, outNode, previousActivation);
                }
                costDerivatives.Add(partialDerivative);
            }
            return costDerivatives.Average();
        }
        public void ApplyGradientDescent(DataPoint[] dataSet)
        {
            for (int layer = 0; layer < nodeCounts.Length - 1; layer++)
            {
                Layer currentLayer = layers[layer];
                for (int outNode = 0; outNode < currentLayer.biases.Length; outNode++)
                {
                    for (int inNode = 0; inNode < currentLayer.weights.GetLength(0); inNode++)
                    {
                        currentLayer.biases[outNode] -= learningStrength * CalculateDataSetCostOverBiasDerivative(dataSet, layer, inNode, outNode);
                        currentLayer.weights[inNode, outNode] -= learningStrength * CalculateDataSetCostOverWeightDerivative(dataSet, layer, inNode, outNode);
                    }
                }
            }
        }
    }
}
