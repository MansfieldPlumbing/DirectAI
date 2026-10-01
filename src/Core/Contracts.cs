using System.Text.Json;

namespace DirectAI;

public sealed record Capability(string Id, string Description);
public sealed record PluginManifest(string Id, string Assembly, string Type, Capability[] Capabilities, string Kind = "assembly", string Script = null);
public sealed record PluginContext(string ModelsDirectory, string OutputDirectory, string PowerShellPath = @"C:\bin\pwsh\pwsh.exe");

public interface IAiPlugin : IDisposable
{
    Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken cancellationToken);
}
