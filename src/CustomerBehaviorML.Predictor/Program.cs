using System;
using System.IO;
using Microsoft.ML;
using CustomerBehaviorML.Core;

namespace CustomerBehaviorML.Predictor
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            string modelPath = Path.Combine("models", "model.zip");

            if (!File.Exists(modelPath))
            {
                Console.WriteLine("Файл моделі не знайдено");
                return;
            }

            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var inputSchema);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<CustomerData, PurchasePrediction>(trainedModel);

            var sampleCustomer = new CustomerData
            {
                Gender = "F",
                Age = "26-35",
                Occupation = 7,
                City_Category = "B",
                Product_Category_1 = 5
            };

            var prediction = predictionEngine.Predict(sampleCustomer);
            Console.WriteLine($"\nПрогнозована сума покупки клієнта: ${prediction.PredictedPurchase:0.00}");
        }
    }
}