using Microsoft.ML.Data;

namespace CovidLiverML
{
    public class PatientData
    {
        [LoadColumn(0)] public string? Cancer { get; set; }
        [LoadColumn(1)] public string? Year { get; set; } 
        [LoadColumn(2)] public string? Month { get; set; }
        [LoadColumn(3)] public string? Bleed { get; set; }
        [LoadColumn(4)] public string? ModePresentation { get; set; }
        [LoadColumn(5)] public float Age { get; set; }
        [LoadColumn(6)] public string? Gender { get; set; }
        [LoadColumn(7)] public string? Etiology { get; set; }
        [LoadColumn(8)] public string? Cirrhosis { get; set; } 
        [LoadColumn(9)] public float Size { get; set; } 
        [LoadColumn(10)] public string? HccTnmStage { get; set; }
        [LoadColumn(11)] public string? HccBclcStage { get; set; }
        [LoadColumn(12)] public string? IccTnmStage { get; set; }
        [LoadColumn(13)] public string? TreatmentGrps { get; set; }
        [LoadColumn(14)] public float SurvivalFromMdm { get; set; }
        
        [LoadColumn(15)] public string? AliveDead { get; set; }

        [LoadColumn(16)] public string? TypeOfIncidentalFinding { get; set; }
        [LoadColumn(17)] public string? SurveillanceProgramme { get; set; }
        [LoadColumn(18)] public string? SurveillanceEffectiveness { get; set; }
        [LoadColumn(19)] public string? ModeOfSurveillanceDetection { get; set; }
        [LoadColumn(20)] public float TimeDiagnosis1stTx { get; set; }
        [LoadColumn(21)] public string? DateIncidentSurveillanceScan { get; set; }
        [LoadColumn(22)] public string? PS { get; set; }
        [LoadColumn(23)] public float TimeMdm1stTreatment { get; set; }
        [LoadColumn(24)] public float TimeDecisionToTreat1stTreatment { get; set; }
        [LoadColumn(25)] public string? PrevKnownCirrhosis { get; set; }
        [LoadColumn(26)] public float MonthsFromLastSurveillance { get; set; }
    }

    public class PatientDataTransformed : PatientData
    {
        public bool Label { get; set; }
    }
}
