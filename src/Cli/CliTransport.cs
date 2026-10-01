using System.Text.Json;
namespace DirectAI;
public sealed class CliTransport(Func<string, JsonElement, CancellationToken, Task<object>> invoke)
{
    public async Task<int> RunAsync(string[] args, TextWriter output, TextWriter error, CancellationToken ct = default)
    {
        try
        {
            string action = args.FirstOrDefault() ?? "capabilities";
            if (action == "invoke") action = args.ElementAtOrDefault(1) ?? throw new ArgumentException("Specify a capability.");
            int index = Array.IndexOf(args, "--request"); string json = index >= 0 ? args.ElementAtOrDefault(index + 1) ?? throw new ArgumentException("Specify request JSON.") : "{}";
            if (json.StartsWith('@')) json = await File.ReadAllTextAsync(json[1..], ct);
            using var request = JsonDocument.Parse(json);
            await output.WriteLineAsync(JsonSerializer.Serialize(await invoke(action, request.RootElement, ct), new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }
        catch (Exception exception) { await error.WriteLineAsync(exception.Message); return 1; }
    }
}
