using System.Text.Json;

namespace DirectAI;

public static class Request
{
    public static string Text(this JsonElement request, string key, string fallback = null) =>
        request.TryGetProperty(key, out var value) && value.ValueKind != JsonValueKind.Null ? value.GetString() : fallback;
    public static int Number(this JsonElement request, string key, int fallback) =>
        request.TryGetProperty(key, out var value) ? value.GetInt32() : fallback;
    public static string ModelPath(string directory, string name)
    {
        string path = Path.GetFullPath(Path.IsPathRooted(name) ? name : Path.Combine(directory, name));
        if (!Directory.Exists(path)) throw new DirectoryNotFoundException($"Model directory not found: {path}");
        return path;
    }
}
