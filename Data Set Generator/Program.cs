using System;
using Newtonsoft.Json;
using Neural_Network;
using System.Drawing;

namespace Data_Set_Generator
{
    public class Program
    {
        public static void CreateParabolaSet(int size)
        {
            DataPoint[] newDataSet = new DataPoint[size * size];
            int index = 0;
            for (double i = 0; i < size; i++)
            {
                for (double j = 0; j < size; j++)
                {
                    DataPoint binaryDataPoint = new DataPoint();
                    binaryDataPoint.inputs = new double[] { i / size, j / size };

                    //Original height is -0.25size^2
                    if (j <= -4d / size * i * (i - size))
                        binaryDataPoint.outputs = new double[] { 1 };
                    else
                        binaryDataPoint.outputs = new double[] { 0 };
                    newDataSet[index] = binaryDataPoint;
                    index++;
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
        public static void CreateImageDataSet(string imagePath)
        { 
            
        }
        static void Main(string[] args)
        {
            CreateParabolaSet(20);
            Console.WriteLine(5 <= -4 / 20f * 5 * (5 - 20));
        }
    }
}