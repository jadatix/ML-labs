using Microsoft.ML.Data;

namespace CustomerBehaviorML.Core
{
    public class CustomerData
    {
        [LoadColumn(2)] public string Gender { get; set; }
        [LoadColumn(3)] public string Age { get; set; }
        [LoadColumn(4)] public float Occupation { get; set; }
        [LoadColumn(5)] public string City_Category { get; set; }
        [LoadColumn(8)] public float Product_Category_1 { get; set; }
        [LoadColumn(11)] public float Purchase { get; set; }
    }
}
