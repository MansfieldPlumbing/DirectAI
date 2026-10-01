using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;

namespace DirectAI;

public sealed class PluginRegistry : IDisposable
{
    private readonly System.Collections.Concurrent.ConcurrentQueue<object> _telemetry = new();
    public object[] Telemetry() => _telemetry.ToArray();
    private sealed class Entry
    {
        public PluginManifest Manifest;
        public string Directory;
        public IAiPlugin Instance;
        public readonly SemaphoreSlim Gate = new(1, 1);
    }

    private readonly List<Entry> _entries = new();
    private readonly PluginContext _context;
    private readonly Dictionary<string, Entry> _capabilities = new(StringComparer.OrdinalIgnoreCase);
    public PluginRegistry(string pluginDirectory, PluginContext context)
    {
        _context = context;
        Directory.CreateDirectory(pluginDirectory);
        foreach (string path in Directory.GetFiles(pluginDirectory, "plugin.json", SearchOption.AllDirectories).Order())
        {
            var manifest = JsonSerializer.Deserialize<PluginManifest>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? throw new InvalidDataException($"Empty plugin manifest: {path}");
            if (string.IsNullOrWhiteSpace(manifest.Id) || manifest.Capabilities is null || manifest.Capabilities.Length == 0)
                throw new InvalidDataException($"Invalid plugin manifest: {path}");
            if (_entries.Any(e => e.Manifest.Id == manifest.Id)) throw new InvalidDataException($"Duplicate plugin ID: {manifest.Id}");
            var entry = new Entry { Manifest = manifest, Directory = Path.GetDirectoryName(path) };
            foreach (var capability in manifest.Capabilities)
                if (!_capabilities.TryAdd(capability.Id, entry)) throw new InvalidDataException($"Duplicate capability: {capability.Id}");
            _entries.Add(entry);
        }
    }

    // Enumeration reads manifests; it never creates a device or loads a model.
    public IReadOnlyList<PluginManifest> Enumerate() => _entries.Select(e => e.Manifest).ToArray();

    public async Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct = default)
    {
        if (!_capabilities.TryGetValue(capability, out var entry)) throw new KeyNotFoundException($"Unknown capability: {capability}");
        await entry.Gate.WaitAsync(ct);
        var timer = System.Diagnostics.Stopwatch.StartNew();
        string invocationId = Guid.NewGuid().ToString("N");
        string failure = null;
        try
        {
            if (entry.Instance is null)
            {
                if (entry.Manifest.Kind == "powershell")
                    entry.Instance = new PowerShellPlugin(entry.Directory, entry.Manifest.Script, _context);
                else
                {
                string path = Path.GetFullPath(Path.Combine(entry.Directory, entry.Manifest.Assembly));
                if (!path.StartsWith(Path.GetFullPath(entry.Directory) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Plugin assembly must reside in its plugin directory.");
                foreach (string dependency in Directory.GetFiles(entry.Directory, "*.dll"))
                    if (!AppDomain.CurrentDomain.GetAssemblies().Any(a => string.Equals(a.GetName().Name, Path.GetFileNameWithoutExtension(dependency), StringComparison.OrdinalIgnoreCase)))
                    {
                        try { AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.GetFullPath(dependency)); }
                        catch (BadImageFormatException) { /* Native dependencies are loaded by their managed owner. */ }
                    }
                var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(path);
                var type = assembly.GetType(entry.Manifest.Type, throwOnError: true);
                entry.Instance = Activator.CreateInstance(type, _context) as IAiPlugin
                    ?? throw new InvalidDataException($"Plugin does not implement {nameof(IAiPlugin)}: {type}");
                }
            }
            return await entry.Instance.InvokeAsync(capability, request, ct);
        }
        catch (Exception error) { failure = error.Message; throw; }
        finally
        {
            timer.Stop();
            _telemetry.Enqueue(new { invocationId, plugin = entry.Manifest.Id, capability, milliseconds = timer.Elapsed.TotalMilliseconds, success = failure == null, error = failure });
            while (_telemetry.Count > 128) _telemetry.TryDequeue(out _);
            entry.Gate.Release();
        }
    }

    public void Dispose()
    {
        foreach (var entry in _entries) { entry.Instance?.Dispose(); entry.Gate.Dispose(); }
    }
}
