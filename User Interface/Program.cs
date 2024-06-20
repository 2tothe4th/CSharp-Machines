using System;
using System.Diagnostics;
using Neural_Network;
using Newtonsoft.Json;

namespace User_Interface
{
    public class Program
    {
        static void Main(string[] args)
        {
            var selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 784, 16, 16, 10 };            

            //https://stackoverflow.com/questions/8308834/creating-a-true-random
            Random random = new Random(DateTime.Now.Millisecond);

            Console.WriteLine("Enter the path to the training dataset");
            DataPoint[] trainingDataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText(Console.ReadLine()));

            Console.WriteLine("Enter the path to the testing dataset");
            DataPoint[] testingDataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText(Console.ReadLine()));

            Console.WriteLine("Type something to import data from the JSON file");
            string jsonQuestionInput = Console.ReadLine();
            if (jsonQuestionInput != "")
                selectedNeuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("../../../NeuralNetworkData.json"));

            Console.WriteLine("Enter the training time in milliseconds");
            //https://stackoverflow.com/questions/11399439/converting-string-to-double-in-c-sharp
            int trainingTime = (int)Convert.ToDouble(Console.ReadLine());

            Console.WriteLine("Enter the mini batch size");
            selectedNeuralNetwork.miniBatchSize = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine("Enter the Neural Network layer sizes");
            //https://stackoverflow.com/questions/823532/apply-function-to-all-elements-of-collection-through-linq
            int[] layerSizes = Console.ReadLine().Replace(" ", "").Split(",").Select(x => Convert.ToInt32(x)).ToArray();

            if (jsonQuestionInput == "")
            {
                selectedNeuralNetwork.nodeCounts = layerSizes;
                selectedNeuralNetwork.CreateLayers();
                selectedNeuralNetwork.RandomizeWeightsAndBiases();
            }


            //https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.elapsed?view=net-8.0
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            while (stopwatch.ElapsedMilliseconds < trainingTime)
            {
                DataPoint[] newDataSet = trainingDataSet.OrderBy(x => random.Next(32768)).ToArray();
                selectedNeuralNetwork.ApplyGradientDescent(trainingDataSet);
                Console.WriteLine("Cost (Training, Testing): " + selectedNeuralNetwork.CalculateCostForDataSet(trainingDataSet) + ", " + selectedNeuralNetwork.CalculateCostForDataSet(testingDataSet));
            }
            stopwatch.Stop();

            /*
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
            Console.WriteLine("Output from " + inputString + ": " + selectedNeuralNetwork.precomputedActivations[3][0]);*/
            using (StreamWriter sw = new StreamWriter("../../../NeuralNetworkData.json"))
            {
                //https://stackoverflow.com/questions/7397207/json-net-error-self-referencing-loop-detected-for-type
                foreach (char charecter in JsonConvert.SerializeObject(selectedNeuralNetwork))
                {
                    sw.Write(charecter);
                }
            }

            //Console.WriteLine("Cost from testing dataset: " + selectedNeuralNetwork.CalculateCostForDataSet(testingDataSet));
            Console.WriteLine("Training Accuracy: " + selectedNeuralNetwork.GetDataSetAccuracy(trainingDataSet));
            Console.WriteLine("Testing Accuracy: " + selectedNeuralNetwork.GetDataSetAccuracy(testingDataSet));
        }
    }
}