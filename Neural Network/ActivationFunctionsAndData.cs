using System;
using System.Data;
using System.Diagnostics;

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
}