using System.Diagnostics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
namespace DirectAI;
public sealed class FaceSwapPlugin(PluginContext context) : IAiPlugin
{
    private FaceSwapEngine _engine; private int _device = -1;
    public Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested(); int device = request.Number("aux", 0);
        if (_engine == null || _device != device) { Dispose(); _engine = new FaceSwapEngine(Path.Combine(context.ModelsDirectory, "inswapper"), device); _device = device; }
        using var image = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("image_base64")));
        using var source = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("source_base64")));
        var timer = Stopwatch.StartNew(); using var result = _engine.SwapFace(image, _engine.ExtractFaceEmbedding(source)); timer.Stop();
        return Task.FromResult(ImageArtifact.Save(result, context.OutputDirectory, capability, request, timer.Elapsed.TotalMilliseconds));
    }
    public void Dispose() { _engine?.Dispose(); _engine = null; }
}
