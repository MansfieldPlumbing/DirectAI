using System.Diagnostics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
namespace DirectAI;
public sealed class SegmentationPlugin(PluginContext context) : IAiPlugin
{
    private SegmentationEngine _engine; private int _device = -1;
    public Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested(); int device = request.Number("aux", 0);
        if (_engine == null || _device != device) { Dispose(); _engine = new SegmentationEngine(Path.Combine(context.ModelsDirectory, "Amuse-Painter-AI", "sam-onnx"), device); _device = device; }
        using var image = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("image_base64")));
        var points = request.TryGetProperty("points", out var value) && value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray().Select(p => (p[0].GetSingle(), p[1].GetSingle(), p[2].GetSingle())).ToArray()
            : new[] { (image.Width / 2f, image.Height / 2f, 1f) };
        var timer = Stopwatch.StartNew();
        Image result;
        if (capability == "images.removebg") result = _engine.RemoveBackground(image, points);
        else { _engine.EncodeImage(image); result = _engine.SegmentPoints(points); }
        using (result) { timer.Stop(); return Task.FromResult(ImageArtifact.Save(result, context.OutputDirectory, capability, request, timer.Elapsed.TotalMilliseconds)); }
    }
    public void Dispose() { _engine?.Dispose(); _engine = null; }
}
