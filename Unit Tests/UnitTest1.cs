using Neural_Network;

namespace Unit_Tests
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            NeuralNetwork selectedNeuralNetwork = new NeuralNetwork();
            selectedNeuralNetwork.nodeCounts = new int[] { 1, 2, 1 };
            selectedNeuralNetwork.CreateLayers();
            /*
            selectedNeuralNetwork.layers[0].weights[0, 0] = 1;
            selectedNeuralNetwork.layers[0].weights[0, 1] = 1;
            selectedNeuralNetwork.layers[1].weights[0, 0] = 2;
            selectedNeuralNetwork.layers[1].weights[1, 0] = 2;*/
            selectedNeuralNetwork.RandomizeWeightsAndBiases();
            DataPoint[] dataSet = new DataPoint[] { new DataPoint(new double[] { 0 }, new double[] { 1 }),
                                                    new DataPoint(new double[] { 1 }, new double[] { 0 })};
            
            bool overallDecrease = true;
            for (int i = 0; i < 10000; i++)
            {
                double originalCost = selectedNeuralNetwork.CalculateCostForDataSet(dataSet);
                selectedNeuralNetwork.ApplyGradientDescent(dataSet);
                if (selectedNeuralNetwork.CalculateCostForDataSet(dataSet) > originalCost)
                {
                    overallDecrease = false;
                    break;
                }
            }
            Assert.IsTrue(overallDecrease);
        }
    }
}