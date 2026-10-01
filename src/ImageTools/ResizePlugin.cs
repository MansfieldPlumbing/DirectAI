using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
namespace DirectAI;
public sealed class ResizePlugin(PluginContext context) : IAiPlugin
{
    public Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        using var image = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("image_base64")));
        int width = request.Number("width", image.Width), height = request.Number("height", image.Height);
        if (width < 1 || height < 1 || width > 8192 || height > 8192) throw new ArgumentException("Invalid resize dimensions.");
        var timer = System.Diagnostics.Stopwatch.StartNew(); using var result = image.Clone(c => c.Resize(width, height)); timer.Stop();
        return Task.FromResult(ImageArtifact.Save(result, context.OutputDirectory, capability, request, timer.Elapsed.TotalMilliseconds));
    }
    public void Dispose() { }
}
