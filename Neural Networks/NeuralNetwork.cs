using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Text.RegularExpressions;
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
        //public double deltaW = 0.01;
        //public double deltaB = 0.01;

        //Activation functions
        Func<double, double> activationFunction = input => ActivationFunctions.Sigmoid(input);
        Func<double, double> activationFunctionDerivative = input => ActivationFunctionDerivatives.Sigmoid(input);

        //Precomputed values
        public List<double[]> precomputedSums = new List<double[]>();
        public List<double[]> precomputedActivations = new List<double[]>();
        //public List<double[]> precomputedWeightGradients = new List<double[]>();
        //public List<double[]> precomputedBiasGradients = new List<double[]>();
        //public List<double[]> precomputedCostOverActivationGradients = new List<double[]>();

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
                return layers[0].CalculateOutputsFromInputs(inputValues, activationFunction);
            return layers[layer].CalculateOutputsFromInputs(CalculateOutputsForLayer(inputValues, layer - 1), activationFunction);
        }
        public double CalculateCostForDataPoint(DataPoint data)
        {
            SetPrecomputedNodes(data.inputs);
            double[] calculatedOutputs = precomputedActivations[nodeCounts.Length - 1];
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
        public void SetPrecomputedNodes(double[] inputs)
        {
            precomputedSums = new List<double[]>();
            precomputedActivations = new List<double[]>();
            precomputedSums.Add(new double[0]);
            precomputedActivations.Add(inputs);
            for (int layer = 0; layer < nodeCounts.Length - 1; layer++)
            {
                precomputedSums.Add(layers[layer].CalculateWeightedSums(precomputedActivations[layer]));
                precomputedActivations.Add(Layer.CalculateOutputsFromWeightedSums(precomputedSums[layer + 1], activationFunction));
            }
        }

        //Learning
        public double[] CalculateDataPointCostOverOutputsGradient(DataPoint data, double[] outputs)
        {
            double[] gradient = new double[outputs.Length];
            for (int outNode = 0; outNode < outputs.Length; outNode++)
            {
                gradient[outNode] = 2 * (outputs[outNode] - data.outputs[outNode]);
            }
            return gradient;
        }
        public void ApplyGradientDescent(DataPoint[] dataSet)
        {
            List<DataPoint> currentDataSample = new List<DataPoint>();
            for (int dataPointIndex = 0; dataPointIndex < dataSet.Length; dataPointIndex++)
            {
                currentDataSample.Add(dataSet[dataPointIndex]);
                //if (dataPointIndex % miniBatchSize != 0 && dataPointIndex != dataSet.Length - 1)
                foreach (DataPoint currentDataPoint in currentDataSample)
                {
                    SetPrecomputedNodes(currentDataPoint.inputs);

                    //https://towardsdatascience.com/part-2-gradient-descent-and-backpropagation-bf90932c066a
                    double[] costOverLastLayerActivationsGradient = CalculateDataPointCostOverOutputsGradient(currentDataPoint, precomputedActivations[nodeCounts.Length - 1]);
                    for (int i = 0; i < costOverLastLayerActivationsGradient.Length; i++)
                    {
                        costOverLastLayerActivationsGradient[i] /= dataSet.Length;
                    }
                    for (int layer = nodeCounts.Length - 2; layer >= 0; layer--)
                    {
                        double[] costOverCurrentLayerActivationsGradient = new double[nodeCounts[layer]];
                        for (int outNode = 0; outNode < nodeCounts[layer + 1]; outNode++)
                        {
                            double activationOverBiasDerivative = activationFunctionDerivative(precomputedSums[layer + 1][outNode]);
                            layers[layer].biases[outNode] -= learningStrength * activationOverBiasDerivative * costOverLastLayerActivationsGradient[outNode];
                            for (int inNode = 0; inNode < nodeCounts[layer]; inNode++)
                            {
                                layers[layer].weights[inNode, outNode] -= learningStrength * activationOverBiasDerivative * precomputedActivations[layer][inNode] * costOverLastLayerActivationsGradient[outNode];
                                //Activation over last activation derivative
                                costOverCurrentLayerActivationsGradient[inNode] += activationOverBiasDerivative * layers[layer].weights[inNode, outNode] * costOverLastLayerActivationsGradient[outNode];
                            }
                        }
                        costOverLastLayerActivationsGradient = costOverCurrentLayerActivationsGradient;
                    }
                }
                currentDataSample.Clear();
            }
        }
    }
}
