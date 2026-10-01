using System.IO.Pipes;
using System.Text;
using System.Text.Json;
namespace DirectAI;
public sealed class NamedPipeTransport(Func<string, JsonElement, CancellationToken, Task<object>> invoke, string name = "directai")
{
    public async Task RunAsync(CancellationToken ct = default)
    {
        while (!ct.IsCancellationRequested)
        {
            var pipe = new NamedPipeServerStream(name, PipeDirection.InOut, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
            try { await pipe.WaitForConnectionAsync(ct); _ = HandleAsync(pipe, ct); }
            catch { pipe.Dispose(); if (ct.IsCancellationRequested) return; throw; }
        }
    }
    private async Task HandleAsync(NamedPipeServerStream pipe, CancellationToken ct)
    {
        try
        {
            using (pipe)
            using (var reader = new StreamReader(pipe, Encoding.UTF8))
            using (var writer = new StreamWriter(pipe, Encoding.UTF8) { AutoFlush = true })
            {
                while (!ct.IsCancellationRequested && pipe.IsConnected)
                {
                    string line = await reader.ReadLineAsync(ct); if (line is null) return;
                    try { using var document = JsonDocument.Parse(line); await writer.WriteLineAsync(JsonSerializer.Serialize(await invoke(document.RootElement.Text("action", "ping"), document.RootElement, ct))); }
                    catch (Exception error) { await writer.WriteLineAsync(JsonSerializer.Serialize(new { error = error.Message })); }
                }
            }
        }
        catch (Exception error) when (error is IOException or OperationCanceledException) { }
    }
}
