using System.IO.Pipes;
using System.Net;
using System.Text;
using System.Text.Json;

namespace DirectAI;

public sealed class MicroserviceHost
{
    private readonly PluginRegistry _registry;
    private readonly string _root;
    private readonly int _port;
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private readonly Dictionary<string, int> _devices = new() { ["textEncoder"] = 0, ["unet"] = 0, ["vaeDecoder"] = 0, ["aux"] = 0 };
    public MicroserviceHost(PluginRegistry registry, string root, int port) { _registry = registry; _root = root; _port = port; }

    private static string Capability(string action) => action switch
    {
        "generate" => "diffusion.generate", "inpaint" => "diffusion.inpaint", "models" => "diffusion.models", "devices" => "directml.devices",
        "erase" => "images.erase", "segment" => "images.segment", "removebg" => "images.removebg", "resize" => "images.resize", "faceswap" => "faces.swap", _ => action
    };

    private async Task<object> InvokeAsync(string action, JsonElement request, CancellationToken ct)
    {
        if (action == "ping") return new { status = "ok" };
        if (action == "capabilities" || action == "plugins") return _registry.Enumerate();
        await _requestGate.WaitAsync(ct);
        try
        {
            if (action == "set_devices")
            {
                var topology = JsonSerializer.SerializeToElement(await _registry.InvokeAsync("directml.devices", request, ct));
                foreach (var property in request.EnumerateObject()) if (_devices.ContainsKey(property.Name))
                {
                    int value = property.Value.GetInt32();
                    if (!topology.GetProperty("devices").EnumerateArray().Any(d => d.GetProperty("DirectMlIndex").GetInt32() == value && d.GetProperty("IsHardware").GetBoolean()))
                        throw new ArgumentException($"Unknown hardware adapter: {value}");
                }
                foreach (var property in request.EnumerateObject()) if (_devices.ContainsKey(property.Name)) _devices[property.Name] = property.Value.GetInt32();
                return new { status = "ok", allocations = _devices };
            }
            var payload = request.EnumerateObject().ToDictionary(p => p.Name, p => (object)p.Value.Clone());
            foreach (var pair in _devices) payload.TryAdd(pair.Key, pair.Value);
            return await _registry.InvokeAsync(Capability(action), JsonSerializer.SerializeToElement(payload), ct);
        }
        finally { _requestGate.Release(); }
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        using var listener = new HttpListener(); listener.Prefixes.Add($"http://127.0.0.1:{_port}/"); listener.Start();
        Console.WriteLine($"DirectAI listening at http://127.0.0.1:{_port}/ and pipe directai");
        var pipe = PipeLoopAsync(ct);
        while (!ct.IsCancellationRequested)
        {
            var context = await listener.GetContextAsync().WaitAsync(ct);
            _ = HandleHttpAsync(context, ct);
        }
        await pipe;
    }

    private async Task HandleHttpAsync(HttpListenerContext context, CancellationToken ct)
    {
        try
        {
            string path = context.Request.Url.AbsolutePath;
            if (context.Request.HttpMethod == "GET" && path == "/")
            {
                context.Response.ContentType = "text/html; charset=utf-8";
                await context.Response.OutputStream.WriteAsync(await File.ReadAllBytesAsync(Path.Combine(AppContext.BaseDirectory, "studio.html"), ct), ct);
                return;
            }
            string action = path.StartsWith("/api/invoke/") ? path[12..] : path.StartsWith("/api/") ? path[5..] : throw new KeyNotFoundException("Unknown endpoint.");
            using var reader = new StreamReader(context.Request.InputStream);
            string body = context.Request.HttpMethod == "POST" ? await reader.ReadToEndAsync(ct) : "{}";
            using var document = JsonDocument.Parse(body);
            if (action == "devices" && context.Request.HttpMethod == "POST") action = "set_devices";
            object result = await InvokeAsync(action, document.RootElement, ct);
            context.Response.ContentType = "application/json";
            await context.Response.OutputStream.WriteAsync(JsonSerializer.SerializeToUtf8Bytes(result), ct);
        }
        catch (Exception error)
        {
            context.Response.StatusCode = error is KeyNotFoundException ? 404 : error is ArgumentException or JsonException or NotSupportedException or DirectoryNotFoundException or FileNotFoundException ? 400 : 500;
            await context.Response.OutputStream.WriteAsync(JsonSerializer.SerializeToUtf8Bytes(new { error = error.Message }), ct);
        }
        finally { context.Response.Close(); }
    }

    private async Task PipeLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            var pipe = new NamedPipeServerStream("directai", PipeDirection.InOut, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
            try { await pipe.WaitForConnectionAsync(ct); _ = HandlePipeAsync(pipe, ct); }
            catch { pipe.Dispose(); throw; }
        }
    }

    private async Task HandlePipeAsync(NamedPipeServerStream pipe, CancellationToken ct)
    {
        using (pipe)
        using (var reader = new StreamReader(pipe, Encoding.UTF8))
        using (var writer = new StreamWriter(pipe, Encoding.UTF8) { AutoFlush = true })
        {
            while (pipe.IsConnected && !ct.IsCancellationRequested)
            {
                string line = await reader.ReadLineAsync(ct); if (line is null) return;
                try
                {
                    using var request = JsonDocument.Parse(line);
                    var result = await InvokeAsync(request.RootElement.Text("action", "ping"), request.RootElement, ct);
                    await writer.WriteLineAsync(JsonSerializer.Serialize(result));
                }
                catch (Exception error) { await writer.WriteLineAsync(JsonSerializer.Serialize(new { error = error.Message })); }
            }
        }
    }
}
