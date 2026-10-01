using System.Security.Cryptography;
using System.Text.Json;

namespace DirectAI.ModelConversion;

public sealed record SourceTensor(string Name, string DataType, long[] Shape, long Offset, long Length);
internal sealed class SafeTensorCheckpoint : IDisposable
{
    private readonly FileStream stream;
    private readonly long dataStart;
    public IReadOnlyDictionary<string, SourceTensor> Tensors { get; }
    public SafeTensorCheckpoint(string path)
    {
        stream = File.OpenRead(path);
        try
        {
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true);
            long headerLength = reader.ReadInt64();
            if (headerLength < 2 || headerLength > 32_000_000 || headerLength > stream.Length - 8)
                throw new InvalidDataException("Invalid safetensors header.");
            dataStart = checked(8 + headerLength);
            using var json = JsonDocument.Parse(reader.ReadBytes((int)headerLength));
            var tensors = new Dictionary<string, SourceTensor>(StringComparer.Ordinal);
            foreach (var entry in json.RootElement.EnumerateObject())
            {
                if (entry.Name == "__metadata__") continue;
                var v = entry.Value;
                string dtype = v.GetProperty("dtype").GetString()!;
                long[] dims = v.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                long[] offsets = v.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                long elements = 1; foreach (long dim in dims) { if (dim <= 0) throw new InvalidDataException("Invalid tensor shape."); elements = checked(elements * dim); }
                int bytes = dtype switch { "F16" or "BF16" => 2, "F32" or "I32" => 4, "I64" => 8, _ => throw new NotSupportedException($"Checkpoint dtype {dtype}.") };
                if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > stream.Length - dataStart || offsets[1] - offsets[0] != checked(elements * bytes))
                    throw new InvalidDataException($"Invalid safetensors extent: {entry.Name}.");
                tensors.Add(entry.Name, new(entry.Name, dtype, dims, offsets[0], offsets[1] - offsets[0]));
            }
            long previousEnd = 0;
            foreach (var tensor in tensors.Values.OrderBy(t => t.Offset))
            {
                if (tensor.Offset != previousEnd) throw new InvalidDataException("Safetensors payload has gaps or overlaps.");
                previousEnd = checked(tensor.Offset + tensor.Length);
            }
            if (previousEnd != stream.Length - dataStart) throw new InvalidDataException("Unaccounted safetensors payload bytes.");
            Tensors = tensors;
        }
        catch { stream.Dispose(); throw; }
    }
    public byte[] Read(SourceTensor tensor)
    {
        if (tensor.Length > int.MaxValue) throw new NotSupportedException("Single tensor exceeds managed transformation limit.");
        byte[] bytes = new byte[(int)tensor.Length];
        stream.Position = checked(dataStart + tensor.Offset); stream.ReadExactly(bytes); return bytes;
    }
    public string Sha256() { stream.Position = 0; return Convert.ToHexString(SHA256.HashData(stream)); }
    public void Dispose() => stream.Dispose();
}
