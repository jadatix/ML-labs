using System;
using System.IO;
using Microsoft.ML;
using CustomerBehaviorML.Core;

namespace CustomerBehaviorML.Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            string dataPath = Path.Combine("data", "train.csv");
            string modelPath = Path.Combine("models", "model.zip");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Файл даних не знайдено за шляхом: {Path.GetFullPath(dataPath)}");
                return;
            }

            IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>(dataPath, hasHeader: true, separatorChar: ',');
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("GenderEncoded", "Gender"),
                    new InputOutputColumnPair("AgeEncoded", "Age"),
                    new InputOutputColumnPair("CityEncoded", "City_Category")
                })
                .Append(mlContext.Transforms.Concatenate("Features", "GenderEncoded", "AgeEncoded", "CityEncoded", "Occupation", "Product_Category_1"))
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Purchase", featureColumnName: "Features"));

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Purchase");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.##}");

            string directory = Path.GetDirectoryName(modelPath);
            if (!Directory.Exists(directory)) Directory.CreateDirectory(directory);

            mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);
            Console.WriteLine($"Модель успішно збережено у файл: {Path.GetFullPath(modelPath)}");
        }
    }
}