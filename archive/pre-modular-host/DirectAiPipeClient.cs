using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace DirectAI;

/// <summary>
/// High-performance Windows Named Pipe client for DirectAI.
/// Communicates over \\.\pipe\{pipeName} with zero network stack overhead.
/// </summary>
public class DirectAiPipeClient : IDisposable
{
    private readonly string _pipeName;
    private NamedPipeClientStream _pipe;
    private StreamReader _reader;
    private StreamWriter _writer;
    private readonly SemaphoreSlim _ioLock = new(1, 1);

    public string PipeName => _pipeName;
    public bool IsConnected => _pipe != null && _pipe.IsConnected;

    public DirectAiPipeClient(string pipeName = "directai")
    {
        _pipeName = pipeName;
    }

    private bool _connected;

    public async Task ConnectAsync(int timeoutMs = 5000, CancellationToken ct = default)
    {
        await _ioLock.WaitAsync(ct);
        try
        {
            if (_connected) return;
            EnsureConnectedCore(timeoutMs);
        }
        finally
        {
            _ioLock.Release();
        }
    }

    private void EnsureConnectedCore(int timeoutMs)
    {
        if (_connected && _pipe != null) return;
        _pipe?.Dispose();
        _pipe = new NamedPipeClientStream(".", _pipeName, PipeDirection.InOut);
        _pipe.Connect(timeoutMs);
        _reader = new StreamReader(_pipe, Encoding.UTF8);
        _writer = new StreamWriter(_pipe, Encoding.UTF8) { AutoFlush = true };
        _connected = true;
    }

    public async Task<JsonDocument> SendRequestAsync(object request, CancellationToken ct = default)
    {
        await _ioLock.WaitAsync(ct);
        try
        {
            if (!_connected)
            {
                EnsureConnectedCore(5000);
            }

            string json = JsonSerializer.Serialize(request);
            await _writer.WriteAsync(json + "\n");
            await _writer.FlushAsync(ct);

            string responseJson = await _reader.ReadLineAsync(ct);
            if (string.IsNullOrEmpty(responseJson))
            {
                _connected = false;
                throw new IOException("Received empty response from DirectAI named pipe server.");
            }

            return JsonDocument.Parse(responseJson);
        }
        catch
        {
            _connected = false;
            throw;
        }
        finally
        {
            _ioLock.Release();
        }
    }

    public async Task<string> PingAsync(CancellationToken ct = default)
    {
        using var doc = await SendRequestAsync(new { action = "ping" }, ct);
        return doc.RootElement.GetProperty("status").GetString();
    }

    public async Task<JsonElement> GetDevicesAsync(CancellationToken ct = default)
    {
        using var doc = await SendRequestAsync(new { action = "devices" }, ct);
        return doc.RootElement.Clone();
    }

    public async Task<string[]> GetModelsAsync(CancellationToken ct = default)
    {
        using var doc = await SendRequestAsync(new { action = "models" }, ct);
        var elem = doc.RootElement.GetProperty("models");
        var list = new List<string>();
        foreach (var item in elem.EnumerateArray())
        {
            list.Add(item.GetString());
        }
        return list.ToArray();
    }

    public async Task<JsonDocument> GenerateAsync(
        string prompt,
        string model = "Quick-LCM-amuse",
        int steps = 4,
        int seed = 42,
        int width = 512,
        int height = 512,
        int candidates = 1,
        CancellationToken ct = default)
    {
        return await SendRequestAsync(new
        {
            action = "generate",
            prompt,
            model,
            steps,
            seed,
            width,
            height,
            candidates
        }, ct);
    }

    public async Task<JsonDocument> InpaintAsync(
        string imageBase64,
        string maskBase64,
        string prompt,
        string model = "Quick-LCM-amuse",
        int steps = 4,
        int seed = 42,
        int candidates = 1,
        CancellationToken ct = default)
    {
        return await SendRequestAsync(new
        {
            action = "inpaint",
            image_base64 = imageBase64,
            mask_base64 = maskBase64,
            prompt,
            model,
            steps,
            seed,
            candidates
        }, ct);
    }

    public async Task<JsonDocument> EraseAsync(
        string imageBase64,
        string maskBase64,
        CancellationToken ct = default)
    {
        return await SendRequestAsync(new
        {
            action = "erase",
            image_base64 = imageBase64,
            mask_base64 = maskBase64
        }, ct);
    }

    public async Task<JsonDocument> SegmentAsync(
        string imageBase64,
        object[] points,
        CancellationToken ct = default)
    {
        return await SendRequestAsync(new
        {
            action = "segment",
            image_base64 = imageBase64,
            points
        }, ct);
    }

    public async Task<JsonDocument> RemoveBgAsync(
        string imageBase64,
        object[] points = null,
        CancellationToken ct = default)
    {
        return await SendRequestAsync(new
        {
            action = "removebg",
            image_base64 = imageBase64,
            points
        }, ct);
    }

    public void Dispose()
    {
        _reader?.Dispose();
        _writer?.Dispose();
        _pipe?.Dispose();
        _ioLock.Dispose();
    }
}
