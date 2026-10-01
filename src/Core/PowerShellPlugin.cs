using System.Diagnostics;
using System.Text.Json;

namespace DirectAI;

internal sealed class PowerShellPlugin : IAiPlugin
{
    private readonly string _script;
    private readonly PluginContext _context;
    public PowerShellPlugin(string directory, string script, PluginContext context)
    {
        _script = Path.GetFullPath(Path.Combine(directory, script));
        if (!_script.StartsWith(Path.GetFullPath(directory) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ||
            !File.Exists(_script) || Path.GetExtension(_script) != ".ps1")
            throw new InvalidDataException("Script plugin must name a PS1 file inside its plugin directory.");
        _context = context;
    }

    public async Task<object> InvokeAsync(string capability, JsonElement request, CancellationToken ct)
    {
        var start = new ProcessStartInfo(_context.PowerShellPath)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true
        };
        foreach (string argument in new[] { "-NoProfile", "-NonInteractive", "-File", _script, "-Capability", capability,
                     "-ModelsDirectory", _context.ModelsDirectory, "-OutputDirectory", _context.OutputDirectory }) start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new IOException("Could not launch script plugin.");
        var stdout = process.StandardOutput.ReadToEndAsync(ct);
        var stderr = process.StandardError.ReadToEndAsync(ct);
        try
        {
            await process.StandardInput.WriteLineAsync(request.GetRawText()); process.StandardInput.Close();
            await process.WaitForExitAsync(ct);
            string output = await stdout, error = await stderr;
            if (process.ExitCode != 0) throw new InvalidOperationException($"Script plugin exited {process.ExitCode}: {error}");
            using var document = JsonDocument.Parse(output);
            return document.RootElement.Clone();
        }
        catch { if (!process.HasExited) process.Kill(entireProcessTree: true); throw; }
    }
    public void Dispose() { }
}
