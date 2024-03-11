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
            bool createNewDataSet = true;
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

            
            while (stopwatch.ElapsedMilliseconds < 5000)
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