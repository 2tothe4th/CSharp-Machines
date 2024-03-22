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
    public enum ActivationFunctionType 
    { 
        Sigmoid,
        ReLU
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
        public static double GetFunctionFromEnum(double input, ActivationFunctionType activationFunction)
        {
            switch (activationFunction)
            { 
                case ActivationFunctionType.Sigmoid:
                    return Sigmoid(input);
                case ActivationFunctionType.ReLU:
                    return ReLU(input);
            }
            return 0;
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
            //https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
            return input >= 0 ? 1 : 0;
        }
        public static double GetFunctionFromEnum(double input, ActivationFunctionType activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunctionType.Sigmoid:
                    return Sigmoid(input);
                case ActivationFunctionType.ReLU:
                    return ReLU(input);
            }
            return 0;
        }
    }    
    internal class Program
    {
        static void Main(string[] args)
        {     
            var selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 2, 2, 2, 1 };
            selectedNeuralNetwork.CreateLayers();

            DataPoint[] dataSet = JsonConvert.DeserializeObject<DataPoint[]>(File.ReadAllText("TrainingDataSet.json"));

            Random random = new Random(DateTime.Now.Millisecond);           

            selectedNeuralNetwork.RandomizeWeightsAndBiases();
            selectedNeuralNetwork.miniBatchSize = 128;
            //selectedNeuralNetwork.activationFunction = ActivationFunctions.ReLU;
            //selectedNeuralNetwork.activationFunctionDerivative = ActivationFunctionDerivatives.ReLU;

            Console.WriteLine("Type something to import data from the JSON file");
            string jsonQuestionInput = Console.ReadLine();
            if (jsonQuestionInput != "")
                selectedNeuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("NeuralNetworkData.json"));

            Console.WriteLine("Enter the training time in milliseconds");

            //https://stackoverflow.com/questions/11399439/converting-string-to-double-in-c-sharp
            int trainingTime = (int)Convert.ToDouble(Console.ReadLine());

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