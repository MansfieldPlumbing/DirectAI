using System.Text.Json;
namespace DirectAI;
public sealed class RequestRouter(PluginRegistry registry)
{
    private readonly object _lock = new();
    private readonly Dictionary<string, int> _devices = new() { ["textEncoder"] = 0, ["unet"] = 0, ["vaeDecoder"] = 0, ["aux"] = 0 };
    public async Task<object> InvokeAsync(string action, JsonElement request, CancellationToken ct)
    {
        if (action == "ping") return new { status = "ok" };
        if (action is "capabilities" or "plugins") return registry.Enumerate();
        if (action == "set_devices")
        {
            var topology = JsonSerializer.SerializeToElement(await registry.InvokeAsync("directml.devices", request, ct));
            foreach (var property in request.EnumerateObject()) if (_devices.ContainsKey(property.Name))
                if (!topology.GetProperty("devices").EnumerateArray().Any(d => d.GetProperty("DirectMlIndex").GetInt32() == property.Value.GetInt32() && d.GetProperty("IsHardware").GetBoolean()))
                    throw new ArgumentException($"Unknown hardware adapter: {property.Value}");
            lock (_lock) { foreach (var property in request.EnumerateObject()) if (_devices.ContainsKey(property.Name)) _devices[property.Name] = property.Value.GetInt32(); return new { status = "ok", allocations = new Dictionary<string, int>(_devices) }; }
        }
        var payload = request.EnumerateObject().ToDictionary(p => p.Name, p => (object)p.Value.Clone());
        lock (_lock) { foreach (var pair in _devices) payload.TryAdd(pair.Key, pair.Value); }
        string capability = action switch { "generate" => "diffusion.generate", "inpaint" => "diffusion.inpaint", "benchmark" => "diffusion.benchmark", "models" => "diffusion.models", "devices" => "directml.devices", "erase" => "images.erase", "segment" => "images.segment", "removebg" => "images.removebg", "resize" => "images.resize", "faceswap" => "faces.swap", _ => action };
        var result = await registry.InvokeAsync(capability, JsonSerializer.SerializeToElement(payload), ct);
        if (action == "devices")
        {
            var json = JsonSerializer.SerializeToElement(result);
            lock (_lock) return new { devices = json.GetProperty("devices").Clone(), allocations = new Dictionary<string, int>(_devices) };
        }
        return result;
    }
}
