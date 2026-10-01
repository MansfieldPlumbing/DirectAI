using System.Diagnostics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DirectAI;

public sealed class DiffusionPlugin : IAiPlugin
{
    private readonly PluginContext _context;
    private DiffusionPipeline _pipeline;
    private string _model;
    private (int te, int unet, int vae) _devices;
    public DiffusionPlugin(PluginContext context) { _context = context; }

    private DiffusionPipeline Pipeline(JsonElement request)
    {
        string model = Request.ModelPath(_context.ModelsDirectory, request.Text("model", "Quick-LCM-amuse"));
        var devices = (request.Number("textEncoder", 0), request.Number("unet", 0), request.Number("vaeDecoder", 0));
        if (_pipeline != null && _model == model && _devices == devices) return _pipeline;
        var topology = DeviceManager.DiscoverTopology();
        foreach (int device in new[] { devices.Item1, devices.Item2, devices.Item3 })
            if (!topology.Devices.Any(d => d.DirectMlIndex == device && d.IsHardware))
                throw new ArgumentException($"Unknown hardware device: {device}");
        _pipeline?.Dispose(); _pipeline = null;
        _pipeline = new DiffusionPipeline(model, devices.Item1, devices.Item2, devices.Item3);
        _model = model; _devices = devices;
        return _pipeline;
    }

    public async Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        if (capability == "directml.devices")
        {
            var topology = DeviceManager.DiscoverTopology();
            return new { devices = topology.Devices, allocations = new { textEncoder = 0, unet = 0, vaeDecoder = 0, aux = 0 } };
        }
        if (capability == "diffusion.models")
            return new { models = Directory.GetDirectories(_context.ModelsDirectory).Select(Path.GetFileName).Order().ToArray(), current = Path.GetFileName(_model) };

        int steps = request.Number("steps", 6), seed = request.Number("seed", 42);
        int width = request.Number("width", 512), height = request.Number("height", 512);
        if (steps < 1 || steps > 50 || width < 8 || height < 8 || width > 2048 || height > 2048 || width % 8 != 0 || height % 8 != 0)
            throw new ArgumentException("Use 1–50 steps and dimensions 8–2048 divisible by eight.");
        string prompt = request.Text("prompt", "a photo");
        var load = Stopwatch.StartNew();
        var pipeline = Pipeline(request); load.Stop();
        bool benchmark = capability == "diffusion.benchmark";
        int warmups = benchmark ? request.Number("warmups", 1) : 0;
        int count = request.Number(benchmark ? "repeats" : "candidates", benchmark ? 3 : 1);
        if (count < 1 || count > 8 || warmups < 0 || warmups > 8) throw new ArgumentException("Use 1–8 outputs and 0–8 warmups.");
        string label = request.Text("label", "benchmark");
        if (label.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || label.Contains('/') || label.Contains('\\')) throw new ArgumentException("Invalid output label.");
        var results = new List<object>(); string primary = null;
        var total = Stopwatch.StartNew();
        for (int i = -warmups; i < count; i++)
        {
            ct.ThrowIfCancellationRequested();
            int candidateSeed = benchmark ? seed : seed + i;
            var timer = Stopwatch.StartNew();
            using Image<Rgba32> image = capability switch
            {
                "diffusion.generate" or "diffusion.benchmark" => await pipeline.GenerateTextToImageAsync(prompt, steps, candidateSeed, width, height, ct: ct),
                "diffusion.inpaint" => await InpaintAsync(pipeline, request, prompt, steps, candidateSeed, ct),
                _ => throw new KeyNotFoundException($"Unknown diffusion capability: {capability}")
            };
            timer.Stop();
            if (i < 0) continue;
            using var stream = new MemoryStream(); image.SaveAsPng(stream); byte[] bytes = stream.ToArray();
            var metadata = new { capability, model = _model, prompt, steps, seed = candidateSeed, width = image.Width, height = image.Height,
                devices = new { textEncoder = _devices.te, unet = _devices.unet, vaeDecoder = _devices.vae },
                inferenceMilliseconds = timer.Elapsed.TotalMilliseconds, loadMilliseconds = load.Elapsed.TotalMilliseconds,
                stages = pipeline.StageTimings, scheduler = "LCM", request = OutputStore.RequestMetadata(request, candidateSeed) };
            string output = OutputStore.SaveBytes(bytes, metadata, Path.Combine(_context.OutputDirectory,
                benchmark ? $"{label}-{i:D2}.png" : $"{DateTime.UtcNow:yyyyMMdd-HHmmss-fff}-{Guid.NewGuid():N}.png"));
            string b64 = Convert.ToBase64String(bytes); primary ??= b64;
            results.Add(new { seed = candidateSeed, image_base64 = b64, output, metadataPath = Path.ChangeExtension(output, ".json"), metadata });
        }
        total.Stop();
        return new { status = "ok", image_base64 = primary, candidates = results, elapsedMs = total.Elapsed.TotalMilliseconds,
            stepMs = total.Elapsed.TotalMilliseconds / (steps * (double)count) };
    }

    private static async Task<Image<Rgba32>> InpaintAsync(DiffusionPipeline pipeline, JsonElement request, string prompt, int steps, int seed, CancellationToken ct)
    {
        using var canvas = Image.Load<Rgba32>(Convert.FromBase64String(request.Text("image_base64")));
        using var mask = Image.Load<L8>(Convert.FromBase64String(request.Text("mask_base64")));
        if (canvas.Size != mask.Size) throw new ArgumentException("Image and mask dimensions must match.");
        return await KritaInpaintEngine.InpaintAsync(pipeline, canvas, mask, prompt, steps, seed, ct: ct);
    }

    public void Dispose() { _pipeline?.Dispose(); }
}
