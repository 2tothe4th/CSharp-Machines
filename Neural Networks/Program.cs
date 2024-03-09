using System;
using System.Data;
using System.Diagnostics;
using Newtonsoft.Json;

namespace Neural_Networks
{
    public class DataPoint
    {
        public double[] inputs;
        public double[] outputs;
    }
    public class ActivationFunctions
    {
        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        public static double ReLU(double input)
        {
            return Math.Max(0, input);
        }
    }
    public class Layer
    {
        public double[,] weights;
        public double[] biases;
        public double[] CalculateOutputs(double[] inputs)
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
                outputs[i] = ActivationFunctions.Sigmoid(outputs[i]);
            }
            return outputs;
        }
    }
    public class NeuralNetwork
    {
        public Layer[] layers;
        public int[] nodeCounts;
        public double learningStrength = 0.5;
        public double weightAndBiasRandomRange = 5;
        public double miniBatchSize = 10;

        //Partial derivative calculations
        public double deltaW = 0.01;
        public double deltaB = 0.01;

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
                return layers[0].CalculateOutputs(inputValues);
            return layers[layer].CalculateOutputs(CalculateOutputsForLayer(inputValues, layer - 1));
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
        public double CalculateCostOverWeightDerivative(DataPoint[] dataSet, int layer, int inNode, int outNode)
        {
            double originalCost = CalculateCostForDataSet(dataSet);
            layers[layer].weights[inNode, outNode] += deltaW;
            double partialDerivative = (CalculateCostForDataSet(dataSet) - originalCost) / deltaW;
            layers[layer].weights[inNode, outNode] -= deltaW;
            return partialDerivative;
        }
        public double CalculateCostOverBiasDerivative(DataPoint[] dataSet, int layer, int outNode)
        {
            double originalCost = CalculateCostForDataSet(dataSet);
            layers[layer].biases[outNode] += deltaB;
            double partialDerivative = (CalculateCostForDataSet(dataSet) - originalCost) / deltaB;
            layers[layer].biases[outNode] -= deltaB;
            return partialDerivative;
        }
        public void ApplyGradientDescent(DataPoint[] dataSet)
        {
            for (int layer = 0; layer < nodeCounts.Length - 1; layer++)
            {
                Layer currentLayer = layers[layer];
                for (int outNode = 0; outNode < currentLayer.weights.GetLength(1); outNode++)
                {
                    currentLayer.biases[outNode] -= learningStrength * CalculateCostOverBiasDerivative(dataSet, layer, outNode);
                    for (int inNode = 0; inNode < currentLayer.weights.GetLength(0); inNode++)
                    {
                        currentLayer.weights[inNode, outNode] -= learningStrength * CalculateCostOverWeightDerivative(dataSet, layer, inNode, outNode);
                    }
                }
            }
        }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            bool createNewDataSet = false;
            if (createNewDataSet)
            {
                DataPoint[] newDataSet = new DataPoint[100];
                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        DataPoint binaryDataPoint = new DataPoint();
                        binaryDataPoint.inputs = new double[] { i, j };
                        if (j <= -0.4d * (i * i - 10 * i))
                            binaryDataPoint.outputs = new double[] { 1 };
                        else
                            binaryDataPoint.outputs = new double[] { 0 };
                        newDataSet[10 * i + j] = binaryDataPoint;
                    }
                }

                //https://learn.microsoft.com/en-us/dotnet/api/system.io.streamwriter?view=net-8.0
                using (StreamWriter sw = new StreamWriter("TrainingDataSet.json"))
                {
                    foreach (char charecter in JsonConvert.SerializeObject(newDataSet))
                    {
                        sw.Write(charecter);
                    }
                }
            }
            var selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 2, 3, 2, 1 };
            selectedNeuralNetwork.CreateLayers();
            DataPoint[] dataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText("TrainingDataSet.json"));

            Random random = new Random(DateTime.Now.Millisecond);           

            selectedNeuralNetwork.RandomizeWeightsAndBiases();

            //https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.elapsed?view=net-8.0
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            while (stopwatch.ElapsedMilliseconds < 10000)
            {
                DataPoint[] newDataSet = dataSet.OrderBy(x => random.Next(32768)).ToArray();
                var dataSetSample = new List<DataPoint>();
                for (int i = 0; i < newDataSet.Length; i++)
                {
                    dataSetSample.Add(newDataSet[i]);
                    if (i % selectedNeuralNetwork.miniBatchSize == selectedNeuralNetwork.miniBatchSize - 1)
                    {
                        selectedNeuralNetwork.ApplyGradientDescent(dataSetSample.ToArray());
                        dataSetSample.Clear();
                        Console.WriteLine(selectedNeuralNetwork.CalculateCostForDataSet(dataSet));
                    }
                }
            }
            Console.WriteLine(selectedNeuralNetwork.CalculateOutputsForLayer(new double[] { 5, 5 }, 3)[0]);
            stopwatch.Stop();
        }
    }
}