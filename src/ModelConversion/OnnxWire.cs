using System.Text;

namespace DirectAI.ModelConversion;

// Keeps unknown ONNX fields intact. This is a wire reader, not a partial graph interpreter.
internal readonly record struct WireField(int Number, int Wire, long Start, long Payload, long Length, long End);
internal static class OnnxWire
{
    public static ulong Varint(BinaryReader reader)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            byte b = reader.ReadByte();
            if (shift == 63 && (b & 0xfe) != 0) throw new InvalidDataException("Protobuf integer overflow.");
            value |= (ulong)(b & 127) << shift;
            if ((b & 128) == 0) return value;
        }
        throw new InvalidDataException("Malformed protobuf integer.");
    }
    public static WireField Field(BinaryReader reader, long limit)
    {
        long start = reader.BaseStream.Position;
        ulong tag = Varint(reader);
        int number = checked((int)(tag >> 3)), wire = (int)(tag & 7);
        if (number == 0) throw new InvalidDataException("Invalid protobuf field.");
        long payload = reader.BaseStream.Position, length;
        switch (wire)
        {
            case 0: Varint(reader); length = reader.BaseStream.Position - payload; break;
            case 1: length = 8; break;
            case 2: length = checked((long)Varint(reader)); payload = reader.BaseStream.Position; break;
            case 5: length = 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
        long end = checked(payload + length);
        if (end > limit) throw new InvalidDataException("Protobuf field exceeds enclosing message.");
        reader.BaseStream.Position = end;
        return new(number, wire, start, payload, length, end);
    }
    public static string Text(BinaryReader reader, WireField field)
    {
        if (field.Wire != 2 || field.Length > 1_000_000) throw new InvalidDataException("Invalid ONNX string.");
        reader.BaseStream.Position = field.Payload;
        string text = Encoding.UTF8.GetString(reader.ReadBytes(checked((int)field.Length)));
        reader.BaseStream.Position = field.End;
        return text;
    }
    public static void Varint(Stream stream, ulong value)
    {
        while (value >= 128) { stream.WriteByte((byte)(value | 128)); value >>= 7; }
        stream.WriteByte((byte)value);
    }
    public static void Bytes(Stream stream, int field, ReadOnlySpan<byte> bytes)
    {
        Varint(stream, (ulong)((field << 3) | 2)); Varint(stream, (ulong)bytes.Length); stream.Write(bytes);
    }
    public static void Text(Stream stream, int field, string text) => Bytes(stream, field, Encoding.UTF8.GetBytes(text));
    public static void Integer(Stream stream, int field, ulong value)
    {
        Varint(stream, (ulong)(field << 3)); Varint(stream, value);
    }
    public static void Copy(BinaryReader reader, Stream destination, long start, long count)
    {
        reader.BaseStream.Position = start;
        byte[] buffer = new byte[(int)Math.Min(count, 65536)];
        while (count > 0)
        {
            int read = reader.Read(buffer, 0, (int)Math.Min(count, buffer.Length));
            if (read == 0) throw new EndOfStreamException();
            destination.Write(buffer, 0, read); count -= read;
        }
    }
}

public sealed record OnnxNode(string Name, string Operation, string[] Inputs, string[] Outputs, Dictionary<string, long> IntegerAttributes);
public sealed record OnnxTensor(string Name, int DataType, long[] Shape, long MessageStart, long MessageLength, long RawStart, long RawLength);
public sealed record OnnxInspection(string Path, long GraphStart, long GraphLength, OnnxNode[] Nodes, OnnxTensor[] Initializers);
public static class OnnxInspector
{
    public static string GraphProgramSha256(string path)
    {
        var graph = Inspect(path);
        using var reader = new BinaryReader(File.OpenRead(path));
        using var program = new MemoryStream();
        reader.BaseStream.Position = graph.GraphStart;
        long end = checked(graph.GraphStart + graph.GraphLength);
        while (reader.BaseStream.Position < end)
        {
            var field = OnnxWire.Field(reader, end);
            if (field.Number != 5) OnnxWire.Copy(reader, program, field.Start, field.End - field.Start);
            reader.BaseStream.Position = field.End;
        }
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(program.ToArray()));
    }
    public static OnnxInspection Inspect(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        var nodes = new List<OnnxNode>(); var tensors = new List<OnnxTensor>();
        long graphStart = 0, graphLength = 0;
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            var field = OnnxWire.Field(reader, reader.BaseStream.Length);
            if (field.Number != 7) continue;
            if (graphLength != 0) throw new InvalidDataException("Multiple ONNX graphs.");
            graphStart = field.Payload; graphLength = field.Length;
            reader.BaseStream.Position = field.Payload;
            while (reader.BaseStream.Position < field.End)
            {
                var item = OnnxWire.Field(reader, field.End);
                if (item.Number == 1) nodes.Add(ReadNode(reader, item));
                if (item.Number == 5) tensors.Add(ReadTensor(reader, item));
                reader.BaseStream.Position = item.End;
            }
        }
        if (graphLength == 0) throw new InvalidDataException("ONNX graph missing.");
        return new(Path.GetFullPath(path), graphStart, graphLength, nodes.ToArray(), tensors.ToArray());
    }
    private static OnnxNode ReadNode(BinaryReader reader, WireField message)
    {
        string name = "", operation = "";
        var inputs = new List<string>(); var outputs = new List<string>(); var attrs = new Dictionary<string, long>();
        reader.BaseStream.Position = message.Payload;
        while (reader.BaseStream.Position < message.End)
        {
            var field = OnnxWire.Field(reader, message.End);
            switch (field.Number)
            {
                case 1: inputs.Add(OnnxWire.Text(reader, field)); break;
                case 2: outputs.Add(OnnxWire.Text(reader, field)); break;
                case 3: name = OnnxWire.Text(reader, field); break;
                case 4: operation = OnnxWire.Text(reader, field); break;
                case 5:
                    string attribute = ""; long? value = null;
                    reader.BaseStream.Position = field.Payload;
                    while (reader.BaseStream.Position < field.End)
                    {
                        var nested = OnnxWire.Field(reader, field.End);
                        if (nested.Number == 1) attribute = OnnxWire.Text(reader, nested);
                        if (nested.Number == 3) { reader.BaseStream.Position = nested.Payload; value = unchecked((long)OnnxWire.Varint(reader)); }
                        reader.BaseStream.Position = nested.End;
                    }
                    if (value.HasValue) attrs[attribute] = value.Value;
                    break;
            }
            reader.BaseStream.Position = field.End;
        }
        return new(name, operation, inputs.ToArray(), outputs.ToArray(), attrs);
    }
    private static OnnxTensor ReadTensor(BinaryReader reader, WireField message)
    {
        string name = ""; int dtype = 0; long raw = 0, length = 0; var dims = new List<long>();
        reader.BaseStream.Position = message.Payload;
        while (reader.BaseStream.Position < message.End)
        {
            var field = OnnxWire.Field(reader, message.End);
            if (field.Number == 1)
            {
                reader.BaseStream.Position = field.Payload;
                if (field.Wire == 0) dims.Add(checked((long)OnnxWire.Varint(reader)));
                else while (reader.BaseStream.Position < field.End) dims.Add(checked((long)OnnxWire.Varint(reader)));
            }
            if (field.Number == 2) { reader.BaseStream.Position = field.Payload; dtype = checked((int)OnnxWire.Varint(reader)); }
            if (field.Number == 8) name = OnnxWire.Text(reader, field);
            if (field.Number == 9) { raw = field.Payload; length = field.Length; }
            reader.BaseStream.Position = field.End;
        }
        return new(name, dtype, dims.ToArray(), message.Payload, message.Length, raw, length);
    }
}
