using System.Net;
using System.Text.Json;
namespace DirectAI;
public sealed class HttpTransport : IDisposable
{
    private readonly HttpListener _listener = new();
    private readonly Func<string, JsonElement, CancellationToken, Task<object>> _invoke;
    private readonly string _studio;
    public HttpTransport(Func<string, JsonElement, CancellationToken, Task<object>> invoke, int port, string studio = null)
    { _invoke = invoke; _studio = studio; _listener.Prefixes.Add($"http://127.0.0.1:{port}/"); }
    public async Task RunAsync(CancellationToken ct = default)
    {
        _listener.Start();
        using var cancellation = ct.Register(_listener.Stop);
        try { while (!ct.IsCancellationRequested) { var context = await _listener.GetContextAsync(); _ = HandleAsync(context, ct); } }
        catch (Exception) when (ct.IsCancellationRequested) { }
    }
    private async Task HandleAsync(HttpListenerContext context, CancellationToken ct)
    {
        try
        {
            string path = context.Request.Url.AbsolutePath;
            if (context.Request.HttpMethod == "GET" && path == "/" && _studio != null)
            { context.Response.ContentType = "text/html; charset=utf-8"; await context.Response.OutputStream.WriteAsync(await File.ReadAllBytesAsync(_studio, ct), ct); return; }
            if (context.Request.HttpMethod is not ("GET" or "POST")) throw new ArgumentException("Use GET or POST.");
            string action = path.StartsWith("/api/invoke/") ? path[12..] : path.StartsWith("/api/") ? path[5..] : throw new KeyNotFoundException("Unknown endpoint.");
            using var reader = new StreamReader(context.Request.InputStream);
            using var document = JsonDocument.Parse(context.Request.HttpMethod == "POST" ? await reader.ReadToEndAsync(ct) : "{}");
            if (action == "devices" && context.Request.HttpMethod == "POST") action = "set_devices";
            object result = await _invoke(action, document.RootElement, ct);
            context.Response.ContentType = "application/json; charset=utf-8";
            await context.Response.OutputStream.WriteAsync(JsonSerializer.SerializeToUtf8Bytes(result), ct);
        }
        catch (Exception error)
        {
            context.Response.StatusCode = error is KeyNotFoundException ? 404 : error is ArgumentException or JsonException or NotSupportedException or IOException ? 400 : 500;
            await context.Response.OutputStream.WriteAsync(JsonSerializer.SerializeToUtf8Bytes(new { error = error.Message }), ct);
        }
        finally { context.Response.Close(); }
    }
    public void Dispose() { _listener.Close(); }
}
