using System;
using System.Data;
using System.Diagnostics;
using Newtonsoft.Json;

namespace Neural_Network
{
    public class DataPoint
    {
        public double[] inputs;
        public double[] outputs;

        public DataPoint() { }
        public DataPoint (double[] inputs, double[] outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;
        }
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
    public class ActivationFunctionDerivatives
    {
        public static double Sigmoid(double input)
        {
            double sigmoidOutput = ActivationFunctions.Sigmoid(input);
            //https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/#:~:text=The%20derivative%20of%20the%20sigmoid%20function%20%CF%83(x)%20is%20the,1%E2%88%92%CF%83(x).
            return sigmoidOutput * (1 - sigmoidOutput);
        }
        public static double ReLU(double input)
        {
            return input >= 0 ? 1 : 0;
        }
    }    
    internal class Program
    {
        static void Main(string[] args)
        {     
            var selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 2, 2, 2, 1 };
            selectedNeuralNetwork.CreateLayers();
            /*
            selectedNeuralNetwork.layers[1].weights[0, 0] = 1;
            selectedNeuralNetwork.layers[1].weights[1, 0] = 1;
            selectedNeuralNetwork.SetPrecomputedNodes(new double[] { 0 });
            Console.WriteLine(selectedNeuralNetwork.precomputedActivations[2][0]);*/

            DataPoint[] dataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText("TrainingDataSet.json"));

            Random random = new Random(DateTime.Now.Millisecond);           

            selectedNeuralNetwork.RandomizeWeightsAndBiases();

            //https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.elapsed?view=net-8.0
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            
            while (stopwatch.ElapsedMilliseconds < 20000)
            {
                DataPoint[] newDataSet = dataSet.OrderBy(x => random.Next(32768)).ToArray();
                selectedNeuralNetwork.ApplyGradientDescent(dataSet);
                Console.WriteLine(selectedNeuralNetwork.CalculateCostForDataSet(newDataSet));
            }            
            stopwatch.Stop();
            selectedNeuralNetwork.SetPrecomputedNodes(new double[] { 5, 5 });
            Console.WriteLine(selectedNeuralNetwork.precomputedActivations[3][0]);
        }
    }
}