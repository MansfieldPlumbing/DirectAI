using System.Text.Json;
using SixLabors.ImageSharp;

namespace DirectAI;
public static class ImageArtifact
{
    public static object Save(Image image, string outputDirectory, string capability, JsonElement request, double elapsedMilliseconds)
    {
        using var stream = new MemoryStream(); image.SaveAsPng(stream); byte[] bytes = stream.ToArray();
        string path = OutputStore.SaveBytes(bytes, new { capability, elapsedMilliseconds, request = OutputStore.RequestMetadata(request) },
            Path.Combine(outputDirectory, $"{DateTime.UtcNow:yyyyMMdd-HHmmss-fff}-{Guid.NewGuid():N}.png"));
        return new { status = "ok", image_base64 = Convert.ToBase64String(bytes), output = path,
            metadataPath = Path.ChangeExtension(path, ".json"), elapsedMs = elapsedMilliseconds };
    }
}
