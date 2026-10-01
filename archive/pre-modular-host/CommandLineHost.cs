using System.Text.Json;
namespace DirectAI;
public static class CommandLineHost
{
public static async Task RunAsync(string[] args)
{


string Get(string name, string fallback) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback; }
string root = Get("--root", Environment.GetEnvironmentVariable("DIRECTAI_ROOT") ?? Environment.CurrentDirectory);
using var registry = new PluginRegistry(Get("--plugins", Path.Combine(root, "app", "plugins")),
    new PluginContext(Get("--models", @"C:\Models\DirectAI"), Get("--output-dir", Path.Combine(root, "output"))));
string command = args.FirstOrDefault() ?? "capabilities";
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
try
{
    if (command == "capabilities" || command == "plugins") Console.WriteLine(JsonSerializer.Serialize(registry.Enumerate(), jsonOptions));
    else if (command == "server") await new MicroserviceHost(registry, root, int.Parse(Get("--port", "5050"))).RunAsync();
    else
    {
        string capability = command switch { "devices" => "directml.devices", "models" => "diffusion.models", "generate" => "diffusion.generate", "benchmark" => "diffusion.benchmark", "invoke" => args.ElementAtOrDefault(1), _ => command };
        string requestJson = Get("--request", "{}");
        if (requestJson.StartsWith('@')) requestJson = File.ReadAllText(requestJson[1..]);
        using var request = JsonDocument.Parse(requestJson);
        var stdout = Console.Out; Console.SetOut(Console.Error);
        object result;
        try { result = await registry.InvokeAsync(capability, request.RootElement); }
        finally { Console.SetOut(stdout); }
        Console.WriteLine(JsonSerializer.Serialize(result, jsonOptions));
    }
}
catch (Exception error) { Console.Error.WriteLine(error.Message); Environment.ExitCode = 1; }

}
}
