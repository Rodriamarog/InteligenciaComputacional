using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BlazorInference.Models;

namespace BlazorInference.Services;

public class OnnxPredictionService : IDisposable
{
    private readonly InferenceSession _session;

    public OnnxPredictionService(IWebHostEnvironment env)
    {
        var modelPath = Path.Combine(env.WebRootPath, "model", "heart_disease.onnx");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"ONNX model not found at '{modelPath}'. " +
                "Copy heart_disease.onnx to wwwroot/model/.");

        _session = new InferenceSession(modelPath);
    }

    /// <summary>
    /// Runs inference on a single patient input.
    /// Returns (hasDisease, probability) where probability is P(disease=1).
    /// </summary>
    public (bool HasDisease, float Probability) Predict(PatientInput input)
    {
        float[] features = input.ToFeatureArray();  // shape [13]

        // Build input tensor [1, 13]
        var tensor = new DenseTensor<float>(features, new[] { 1, 13 });

        var inputName = _session.InputMetadata.Keys.First();
        using var results = _session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        });

        var allResults = results.ToList();

        // Output 0: predicted label (long or int64)
        long label = allResults[0].AsTensor<long>().First();

        // Output 1: seq(map(int64, tensor(float))) — skl2onnx RandomForest format.
        // Each sequence element is a NamedOnnxValue wrapping a map; cast the map
        // values to float via AsDictionary<long, float>().
        float prob = 0f;
        var probSeq = allResults[1].AsEnumerable<NamedOnnxValue>().ToList();
        if (probSeq.Count > 0)
        {
            var dict = probSeq[0].AsDictionary<long, float>();
            dict.TryGetValue(1L, out prob);
        }

        return (label == 1L, prob);
    }

    public void Dispose() => _session?.Dispose();
}
