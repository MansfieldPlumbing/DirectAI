using System.Security.Cryptography;
using System.Text.Json;
using SixLabors.ImageSharp;

namespace DirectAI;

public static class OutputStore
{
    public static string Save(Image image, object metadata, string path = null)
    {
        using var stream = new MemoryStream();
        image.SaveAsPng(stream);
        return SaveBytes(stream.ToArray(), metadata, path);
    }

    public static string SaveBytes(byte[] bytes, object metadata, string path = null)
    {
        string folder = Environment.GetEnvironmentVariable("DIRECTAI_OUTPUT_DIR")
            ?? Path.Combine(Environment.CurrentDirectory, "output");
        path ??= $"{DateTime.UtcNow:yyyyMMdd-HHmmss-fff}-{Guid.NewGuid():N}.png";
        if (!Path.IsPathRooted(path) && string.IsNullOrEmpty(Path.GetDirectoryName(path)))
            path = Path.Combine(folder, path);
        path = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, bytes);
        File.WriteAllText(Path.ChangeExtension(path, ".json"), JsonSerializer.Serialize(new
        {
            schemaVersion = 1,
            createdUtc = DateTime.UtcNow,
            output = Path.GetFileName(path),
            sha256 = Convert.ToHexString(SHA256.HashData(bytes)),
            runtime = Environment.Version.ToString(),
            metadata
        }, new JsonSerializerOptions { WriteIndented = true }));
        return path;
    }

    public static object RequestMetadata(JsonElement request, int? candidateSeed = null)
    {
        var fields = new Dictionary<string, object>();
        foreach (var property in request.EnumerateObject())
        {
            if (property.Name.EndsWith("base64", StringComparison.OrdinalIgnoreCase))
            {
                byte[] input = Convert.FromBase64String(property.Value.GetString());
                fields[property.Name] = new { bytes = input.Length, sha256 = Convert.ToHexString(SHA256.HashData(input)) };
            }
            else fields[property.Name] = property.Value.Clone();
        }
        if (candidateSeed.HasValue) fields["seed"] = candidateSeed.Value;
        return fields;
    }
}
