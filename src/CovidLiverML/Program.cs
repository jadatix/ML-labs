using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace CovidLiverML
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            string dataPath = Path.Combine("data", "covid-liver.csv");
            string modelPath = Path.Combine("models", "cancer_model.zip");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine("Файл не знайдено: data/covid-liver.csv");
                return;
            }

            IDataView rawData = mlContext.Data.LoadFromTextFile<PatientData>(dataPath, hasHeader: true, separatorChar: ',');

            var pipelinePrep = mlContext.Transforms.CustomMapping<PatientData, PatientDataTransformed>(
                new CancerMappingFactory().GetMapping(),
                contractName: "CancerMapping")

                .Append(mlContext.Transforms.ReplaceMissingValues("Age", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("Size", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("SurvivalFromMdm", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("TimeDiagnosis1stTx", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("TimeMdm1stTreatment", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("TimeDecisionToTreat1stTreatment", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues("MonthsFromLastSurveillance", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("AliveDeadEncoded", nameof(PatientData.AliveDead)),
                    new InputOutputColumnPair("YearEncoded", nameof(PatientData.Year)),
                    new InputOutputColumnPair("MonthEncoded", nameof(PatientData.Month)),
                    new InputOutputColumnPair("BleedEncoded", nameof(PatientData.Bleed)),
                    new InputOutputColumnPair("ModePresentationEncoded", nameof(PatientData.ModePresentation)),
                    new InputOutputColumnPair("GenderEncoded", nameof(PatientData.Gender)),
                    new InputOutputColumnPair("EtiologyEncoded", nameof(PatientData.Etiology)),
                    new InputOutputColumnPair("CirrhosisEncoded", nameof(PatientData.Cirrhosis)),
                    new InputOutputColumnPair("HccTnmStageEncoded", nameof(PatientData.HccTnmStage)),
                    new InputOutputColumnPair("HccBclcStageEncoded", nameof(PatientData.HccBclcStage)),
                    new InputOutputColumnPair("IccTnmStageEncoded", nameof(PatientData.IccTnmStage)),
                    new InputOutputColumnPair("TreatmentGrpsEncoded", nameof(PatientData.TreatmentGrps)),
                    new InputOutputColumnPair("TypeOfIncidentalFindingEncoded", nameof(PatientData.TypeOfIncidentalFinding)),
                    new InputOutputColumnPair("SurveillanceProgrammeEncoded", nameof(PatientData.SurveillanceProgramme)),
                    new InputOutputColumnPair("SurveillanceEffectivenessEncoded", nameof(PatientData.SurveillanceEffectiveness)),
                    new InputOutputColumnPair("ModeOfSurveillanceDetectionEncoded", nameof(PatientData.ModeOfSurveillanceDetection)),
                    new InputOutputColumnPair("DateIncidentSurveillanceScanEncoded", nameof(PatientData.DateIncidentSurveillanceScan)),
                    new InputOutputColumnPair("PSEncoded", nameof(PatientData.PS)),
                    new InputOutputColumnPair("PrevKnownCirrhosisEncoded", nameof(PatientData.PrevKnownCirrhosis))
                }))

                .Append(mlContext.Transforms.Concatenate("Features",
                    "AliveDeadEncoded", "YearEncoded", "MonthEncoded", "BleedEncoded", "ModePresentationEncoded",
                    "GenderEncoded", "EtiologyEncoded", "CirrhosisEncoded", "HccTnmStageEncoded", "HccBclcStageEncoded",
                    "IccTnmStageEncoded", "TreatmentGrpsEncoded", "TypeOfIncidentalFindingEncoded", "SurveillanceProgrammeEncoded",
                    "SurveillanceEffectivenessEncoded", "ModeOfSurveillanceDetectionEncoded", "DateIncidentSurveillanceScanEncoded",
                    "PSEncoded", "PrevKnownCirrhosisEncoded",
                    nameof(PatientData.Age), nameof(PatientData.Size), nameof(PatientData.SurvivalFromMdm),
                    nameof(PatientData.TimeDiagnosis1stTx), nameof(PatientData.TimeMdm1stTreatment),
                    nameof(PatientData.TimeDecisionToTreat1stTreatment), nameof(PatientData.MonthsFromLastSurveillance)));

            var prepModel = pipelinePrep.Fit(rawData);
            var transformedData = prepModel.Transform(rawData);
            var split = mlContext.Data.TrainTestSplit(transformedData, testFraction: 0.2, seed: 0);

            var candidates = new (string Name, IEstimator<ITransformer> Trainer)[]
            {
                ("FastTree Binary", mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features")),
                ("SDCA Logistic Regression", mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features")),
                ("LBFGS Logistic Regression", mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"))
            };

            Console.WriteLine($"{"Алгоритм",-28} | {"Accuracy",-10} | {"Precision",-10} | {"Recall",-10} | {"F1-Score",-10}");
            Console.WriteLine(new string('-', 78));

            var results = candidates
                .Select(candidate =>
                {
                    var model = candidate.Trainer.Fit(split.TrainSet);

                    var metrics = mlContext.BinaryClassification.Evaluate(
                        model.Transform(split.TestSet),
                        labelColumnName: "Label");

                    PrintRow(candidate.Name, metrics);

                    return new
                    {
                        candidate.Name,
                        Model = model,
                        Metrics = metrics
                    };
                })
                .ToList();

            var bestResult = results.MaxBy(x => (x.Metrics.F1Score * 0.7) + (x.Metrics.Accuracy * 0.3))!;

            Console.WriteLine($"\nПереможець за точністю: {bestResult.Name}");

            var fullModel = prepModel.Append(bestResult.Model);

            string modelDirectory = Path.GetDirectoryName(modelPath);
            if (!string.IsNullOrEmpty(modelDirectory) && !Directory.Exists(modelDirectory))
            {
                Directory.CreateDirectory(modelDirectory);
            }

            mlContext.Model.Save(fullModel, rawData.Schema, modelPath);
            Console.WriteLine($"Збережено модель у: {modelPath}");

            Console.WriteLine("\n=== Тестовий прогноз завантаженою моделлю ===");

            ITransformer loadedModel = mlContext.Model.Load(modelPath, out _);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PatientData, PatientPrediction>(loadedModel);

            var sample = new PatientData
            {
                Cancer = "Y",
                Year = "Prepandemic",
                Month = "1",
                Bleed = "N",
                ModePresentation = "Surveillance",
                Age = 68f,
                Gender = "M",
                Etiology = "NAFLD",
                Cirrhosis = "Y",
                Size = 22f,
                HccTnmStage = "II",
                HccBclcStage = "A",
                IccTnmStage = "NA",
                TreatmentGrps = "Ablation",
                SurvivalFromMdm = 32.73f,
                TypeOfIncidentalFinding = "NA",
                SurveillanceProgramme = "Y",
                SurveillanceEffectiveness = "Consistent",
                ModeOfSurveillanceDetection = "US",
                TimeDiagnosis1stTx = 0.47f,
                AliveDead = "Alive",
                DateIncidentSurveillanceScan = "NA",
                PS = "0",
                TimeMdm1stTreatment = 0.7f,
                TimeDecisionToTreat1stTreatment = 0f,
                PrevKnownCirrhosis = "Y",
                MonthsFromLastSurveillance = 7.33f
            };

            var pred = predictionEngine.Predict(sample);
            Console.WriteLine($"Пацієнт: Вік {sample.Age}, Стать {sample.Gender}, Живий {sample.AliveDead}, Лікування {sample.TreatmentGrps}");
            Console.WriteLine($"Прогноз раку: {(pred.PredictedLabel ? "Y" : "N")}");
            Console.WriteLine($"Імовірність: {pred.Probability:P2}");
        }

        static void PrintRow(string name, BinaryClassificationMetrics m)
        {
            Console.WriteLine($"{name,-28} | {m.Accuracy,10:P2} | {m.PositivePrecision,10:P2} | {m.PositiveRecall,10:P2} | {m.F1Score,10:P2}");
        }
    }

    // CustomMapping для збереження, без нього була помилка при використанні моделі 
    [CustomMappingFactoryAttribute("CancerMapping")]
    public class CancerMappingFactory : CustomMappingFactory<PatientData, PatientDataTransformed>
    {
        public override Action<PatientData, PatientDataTransformed> GetMapping()
        {
            return (input, output) =>
            {
                output.AliveDead = input.AliveDead;
                output.Year = input.Year;
                output.Month = input.Month;
                output.Bleed = input.Bleed;
                output.ModePresentation = input.ModePresentation;
                output.Age = input.Age;
                output.Gender = input.Gender;
                output.Etiology = input.Etiology;
                output.Cirrhosis = input.Cirrhosis;
                output.Size = input.Size;
                output.HccTnmStage = input.HccTnmStage;
                output.HccBclcStage = input.HccBclcStage;
                output.IccTnmStage = input.IccTnmStage;
                output.TreatmentGrps = input.TreatmentGrps;
                output.SurvivalFromMdm = input.SurvivalFromMdm;
                output.TypeOfIncidentalFinding = input.TypeOfIncidentalFinding;
                output.SurveillanceProgramme = input.SurveillanceProgramme;
                output.SurveillanceEffectiveness = input.SurveillanceEffectiveness;
                output.ModeOfSurveillanceDetection = input.ModeOfSurveillanceDetection;
                output.TimeDiagnosis1stTx = input.TimeDiagnosis1stTx;
                output.DateIncidentSurveillanceScan = input.DateIncidentSurveillanceScan;
                output.PS = input.PS;
                output.TimeMdm1stTreatment = input.TimeMdm1stTreatment;
                output.TimeDecisionToTreat1stTreatment = input.TimeDecisionToTreat1stTreatment;
                output.PrevKnownCirrhosis = input.PrevKnownCirrhosis;
                output.MonthsFromLastSurveillance = input.MonthsFromLastSurveillance;

                output.Label = input.Cancer != null && input.Cancer.Equals("\"Y\"", StringComparison.OrdinalIgnoreCase);
            };
        }
    }
}