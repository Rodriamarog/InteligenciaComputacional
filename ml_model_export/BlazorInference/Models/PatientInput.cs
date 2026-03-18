using System.ComponentModel.DataAnnotations;

namespace BlazorInference.Models;

public class PatientInput
{
    // ── Numerical features ──────────────────────────────────────────────────

    [Required]
    [Range(1, 120, ErrorMessage = "Age must be between 1 and 120.")]
    [Display(Name = "Age")]
    public int Age { get; set; } = 50;

    [Required]
    [Range(50, 250, ErrorMessage = "Resting blood pressure must be 50–250 mmHg.")]
    [Display(Name = "Resting Blood Pressure (mmHg)")]
    public int Trestbps { get; set; } = 120;

    [Required]
    [Range(100, 600, ErrorMessage = "Cholesterol must be 100–600 mg/dl.")]
    [Display(Name = "Serum Cholesterol (mg/dl)")]
    public int Chol { get; set; } = 200;

    [Required]
    [Range(60, 250, ErrorMessage = "Max heart rate must be 60–250 bpm.")]
    [Display(Name = "Max Heart Rate Achieved (bpm)")]
    public int Thalach { get; set; } = 150;

    [Required]
    [Range(0.0, 10.0, ErrorMessage = "ST depression must be 0.0–10.0.")]
    [Display(Name = "ST Depression (oldpeak)")]
    public double Oldpeak { get; set; } = 0.0;

    // ── Categorical features ─────────────────────────────────────────────────

    [Required]
    [Range(0, 1, ErrorMessage = "Sex must be 0 or 1.")]
    [Display(Name = "Sex")]
    public int Sex { get; set; } = 1;   // 0 = Female, 1 = Male

    [Required]
    [Range(0, 3, ErrorMessage = "Chest pain type must be 0–3.")]
    [Display(Name = "Chest Pain Type")]
    public int Cp { get; set; } = 0;    // 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic

    [Required]
    [Range(0, 1, ErrorMessage = "Fasting blood sugar must be 0 or 1.")]
    [Display(Name = "Fasting Blood Sugar > 120 mg/dl")]
    public int Fbs { get; set; } = 0;   // 0 = No, 1 = Yes

    [Required]
    [Range(0, 2, ErrorMessage = "Resting ECG must be 0–2.")]
    [Display(Name = "Resting ECG Result")]
    public int Restecg { get; set; } = 0; // 0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy

    [Required]
    [Range(0, 1, ErrorMessage = "Exercise induced angina must be 0 or 1.")]
    [Display(Name = "Exercise Induced Angina")]
    public int Exang { get; set; } = 0;  // 0 = No, 1 = Yes

    [Required]
    [Range(0, 2, ErrorMessage = "Slope must be 0–2.")]
    [Display(Name = "Slope of Peak Exercise ST Segment")]
    public int Slope { get; set; } = 1;  // 0=Upsloping, 1=Flat, 2=Downsloping

    [Required]
    [Range(0, 4, ErrorMessage = "Number of major vessels must be 0–4.")]
    [Display(Name = "Major Vessels Colored by Fluoroscopy (ca)")]
    public int Ca { get; set; } = 0;

    [Required]
    [Range(0, 3, ErrorMessage = "Thal must be 0–3.")]
    [Display(Name = "Thalassemia (thal)")]
    public int Thal { get; set; } = 1;  // 0=Normal, 1=Fixed defect, 2=Reversible defect, 3=?

    /// <summary>
    /// Returns feature values in the exact column order used during training:
    /// ['age','trestbps','chol','thalach','oldpeak','sex','cp','fbs','restecg','exang','slope','ca','thal']
    /// </summary>
    public float[] ToFeatureArray() => new float[]
    {
        Age, Trestbps, Chol, Thalach, (float)Oldpeak,
        Sex, Cp, Fbs, Restecg, Exang, Slope, Ca, Thal
    };
}
