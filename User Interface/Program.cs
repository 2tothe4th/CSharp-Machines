using System;
using System.Diagnostics;
using Neural_Network;
using Newtonsoft.Json;

namespace MyApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            var selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 2, 2, 2, 1 };
            selectedNeuralNetwork.CreateLayers();

            Random random = new Random(DateTime.Now.Millisecond);

            selectedNeuralNetwork.RandomizeWeightsAndBiases();

            Console.WriteLine("Enter the path to the dataset");
            DataPoint[] dataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText(Console.ReadLine()));

            Console.WriteLine("Type something to import data from the JSON file");
            string jsonQuestionInput = Console.ReadLine();
            if (jsonQuestionInput != "")
                selectedNeuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("NeuralNetworkData.json"));

            Console.WriteLine("Enter the training time in milliseconds");
            //https://stackoverflow.com/questions/11399439/converting-string-to-double-in-c-sharp
            int trainingTime = (int)Convert.ToDouble(Console.ReadLine());

            Console.WriteLine("Enter the mini batch size");
            selectedNeuralNetwork.miniBatchSize = Convert.ToInt32(Console.ReadLine());

            //https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.elapsed?view=net-8.0
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            while (stopwatch.ElapsedMilliseconds < trainingTime)
            {
                DataPoint[] newDataSet = dataSet.OrderBy(x => random.Next(32768)).ToArray();
                selectedNeuralNetwork.ApplyGradientDescent(dataSet);
                Console.WriteLine("Cost: " + selectedNeuralNetwork.CalculateCostForDataSet(newDataSet));
            }
            stopwatch.Stop();

            Console.WriteLine("Input a test value");
            string[] stringInputs = Console.ReadLine().Replace(" ", "").Split(",");
            double[] inputs = new double[stringInputs.Length];
            for (int i = 0; i < stringInputs.Length; i++)
            {
                inputs[i] = Convert.ToDouble(stringInputs[i]);
            }
            selectedNeuralNetwork.SetPrecomputedNodes(inputs);
            string inputString = "";
            for (int i = 0; i < inputs.Length - 1; i++)
            {
                inputString += inputs[i] + ", ";
            }
            inputString += inputs[inputs.Length - 1];
            Console.WriteLine("Output from " + inputString + ": " + selectedNeuralNetwork.precomputedActivations[3][0]);
            using (StreamWriter sw = new StreamWriter("NeuralNetworkData.json"))
            {
                //https://stackoverflow.com/questions/7397207/json-net-error-self-referencing-loop-detected-for-type
                foreach (char charecter in JsonConvert.SerializeObject(selectedNeuralNetwork))
                {
                    sw.Write(charecter);
                }
            }
        }
    }
}