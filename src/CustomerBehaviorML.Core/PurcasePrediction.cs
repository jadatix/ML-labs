using Microsoft.ML.Data;

namespace CustomerBehaviorML.Core
{
    public class PurchasePrediction
    {
        [ColumnName("Score")]
        public float PredictedPurchase { get; set; }
    }
}
