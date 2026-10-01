using System.Diagnostics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
namespace DirectAI;
public sealed class ErasePlugin(PluginContext context) : IAiPlugin
{
    private GenerativeEraseEngine _engine; private int _device = -1;
    public Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested(); int device = request.Number("aux", 0);
        if (_engine == null || _device != device) { Dispose(); _engine = new GenerativeEraseEngine(Path.Combine(context.ModelsDirectory, "Amuse-Painter-AI", "migan", "migan_pipeline_v2.onnx"), device); _device = device; }
        using var image = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("image_base64")));
        using var mask = Image.Load<L8>(Convert.FromBase64String(request.Text("mask_base64")));
        if (image.Size != mask.Size) throw new ArgumentException("Image and mask dimensions must match.");
        var timer = Stopwatch.StartNew(); using var result = _engine.Erase(image, mask); timer.Stop();
        return Task.FromResult(ImageArtifact.Save(result, context.OutputDirectory, capability, request, timer.Elapsed.TotalMilliseconds));
    }
    public void Dispose() { _engine?.Dispose(); _engine = null; }
}
