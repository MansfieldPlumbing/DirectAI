using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace DirectAI.ModelConversion;

public static class CheckpointConverter
{
    public static object Convert(string checkpointPath, string templateDirectory, string outputDirectory, string architecture = "sd15")
    {
        if (architecture != "sd15") throw new NotSupportedException("Only the SD1.5 recipe is admitted; SDXL requires its own validated dual-encoder recipe.");
        string destination = Path.GetFullPath(outputDirectory);
        if (Directory.Exists(destination) && Directory.EnumerateFileSystemEntries(destination).Any())
            throw new IOException("Output directory must be new or empty.");
        string[] stages = ["text_encoder", "unet", "vae_decoder", "vae_encoder"];
        foreach (string stage in stages)
            if (!File.Exists(Path.Combine(templateDirectory, stage, "model.onnx"))) throw new FileNotFoundException($"Graph recipe missing: {stage}.");
        if (!Directory.Exists(Path.Combine(templateDirectory, "tokenizer"))) throw new DirectoryNotFoundException("CLIP tokenizer recipe missing.");
        using var checkpoint = new SafeTensorCheckpoint(checkpointPath);
        var plans = new List<(string Stage, OnnxInspection Graph, Dictionary<string, WeightBinding> Bindings)>();
        foreach (string stage in stages)
        {
            var graph = OnnxInspector.Inspect(Path.Combine(templateDirectory, stage, "model.onnx"));
            var bindings = new Dictionary<string, WeightBinding>(StringComparer.Ordinal);
            foreach (var tensor in graph.Initializers.Where(t => t.DataType is 1 or 10))
            {
                var binding = CheckpointMappings.Resolve(stage, tensor, graph, false, checkpoint.Tensors);
                foreach (string key in binding.Sources)
                    if (!checkpoint.Tensors.ContainsKey(key)) throw new InvalidDataException($"Missing source weight for {stage}:{tensor.Name}: {key}.");
                bindings.Add(tensor.Name, binding);
            }
            plans.Add((stage, graph, bindings));
        }
        var plannedSources = plans.SelectMany(p => p.Bindings.Values).SelectMany(b => b.Sources).ToHashSet(StringComparer.Ordinal);
        string[] allowedUnused = ["alphas_cumprod", "alphas_cumprod_prev", "betas", "cond_stage_model.transformer.text_model.embeddings.position_ids",
            "log_one_minus_alphas_cumprod", "model_ema.decay", "model_ema.num_updates", "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2", "posterior_variance", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"];
        string[] unresolved = checkpoint.Tensors.Keys.Except(plannedSources).Except(allowedUnused).ToArray();
        if (unresolved.Length != 0) throw new InvalidDataException("Unmapped checkpoint tensors: " + string.Join(", ", unresolved));
        string checkpointHash = checkpoint.Sha256();
        Directory.CreateDirectory(destination);
        var reports = new List<object>(); var used = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            foreach (var plan in plans)
            {
                string stage = plan.Stage, template = plan.Graph.Path;
                var graph = plan.Graph; var bindings = plan.Bindings;
                string stageDirectory = Path.Combine(destination, stage);
                Directory.CreateDirectory(stageDirectory);
                string modelPath = Path.Combine(stageDirectory, "model.onnx"), dataPath = modelPath + ".data";
                using var input = new BinaryReader(File.OpenRead(template));
                using var weights = File.Create(dataPath);
                using var graphOutput = new MemoryStream();
                var tensorReports = new List<object>();
                input.BaseStream.Position = graph.GraphStart;
                var tensors = graph.Initializers.ToDictionary(t => t.MessageStart);
                long graphEnd = checked(graph.GraphStart + graph.GraphLength);
                while (input.BaseStream.Position < graphEnd)
                {
                    var field = OnnxWire.Field(input, graphEnd);
                    if (field.Number == 5 && tensors.TryGetValue(field.Payload, out var tensor) && bindings.TryGetValue(tensor.Name, out var binding))
                    {
                        byte[] payload = Transform(checkpoint, tensor, binding);
                        string payloadHash = System.Convert.ToHexString(SHA256.HashData(payload));
                        bool? templateMatch = null;
                        object? templateDifference = null;
                        if (tensor.RawLength == payload.Length)
                        {
                            input.BaseStream.Position = tensor.RawStart;
                            byte[] original = input.ReadBytes(payload.Length);
                            templateMatch = payload.AsSpan().SequenceEqual(original);
                            templateDifference = Difference(payload, original, tensor.DataType);
                        }
                        long padding = (64 - weights.Position % 64) % 64;
                        for (long p = 0; p < padding; p++) weights.WriteByte(0);
                        long offset = weights.Position; weights.Write(payload);
                        using var externalTensor = new MemoryStream();
                        input.BaseStream.Position = tensor.MessageStart;
                        long tensorEnd = checked(tensor.MessageStart + tensor.MessageLength);
                        while (input.BaseStream.Position < tensorEnd)
                        {
                            var attribute = OnnxWire.Field(input, tensorEnd);
                            if (attribute.Number is not (4 or 5 or 6 or 7 or 9 or 10 or 11 or 13 or 14))
                                OnnxWire.Copy(input, externalTensor, attribute.Start, attribute.End - attribute.Start);
                            input.BaseStream.Position = attribute.End;
                        }
                        External(externalTensor, "location", "model.onnx.data");
                        External(externalTensor, "offset", offset.ToString(System.Globalization.CultureInfo.InvariantCulture));
                        External(externalTensor, "length", payload.Length.ToString(System.Globalization.CultureInfo.InvariantCulture));
                        OnnxWire.Integer(externalTensor, 14, 1);
                        OnnxWire.Bytes(graphOutput, 5, externalTensor.ToArray());
                        foreach (string key in binding.Sources) used.Add(key);
                        tensorReports.Add(new { initializer = tensor.Name, tensor.Shape, tensor.DataType, sources = binding.Sources,
                            transform = binding.Transform, heads = binding.Heads, bytes = payload.Length, offset, sha256 = payloadHash,
                            identicalToTemplatePayload = templateMatch, templateDifference });
                    }
                    else OnnxWire.Copy(input, graphOutput, field.Start, field.End - field.Start);
                    input.BaseStream.Position = field.End;
                }
                weights.Flush(true);
                weights.Dispose();
                using (var output = File.Create(modelPath))
                {
                    input.BaseStream.Position = 0;
                    while (input.BaseStream.Position < input.BaseStream.Length)
                    {
                        var field = OnnxWire.Field(input, input.BaseStream.Length);
                        if (field.Number == 7) OnnxWire.Bytes(output, 7, graphOutput.ToArray());
                        else OnnxWire.Copy(input, output, field.Start, field.End - field.Start);
                        input.BaseStream.Position = field.End;
                    }
                    Metadata(output, "DirectAI.checkpoint.sha256", checkpointHash);
                    Metadata(output, "DirectAI.recipe.architecture", architecture);
                }
                var report = new { stage, template = Path.GetFullPath(template), templateSha256 = HashFile(template),
                    model = modelPath, modelSha256 = HashFile(modelPath), weights = dataPath, weightsSha256 = HashFile(dataPath),
                    boundInitializers = bindings.Count, sourceWeights = tensorReports };
                File.WriteAllText(Path.Combine(stageDirectory, "conversion.json"), JsonSerializer.Serialize(report, Options));
                reports.Add(report);
                Console.Error.WriteLine($"Converted {stage}: {bindings.Count} bound initializers.");
            }
            string tokenizer = Path.Combine(templateDirectory, "tokenizer");
            if (!Directory.Exists(tokenizer)) throw new DirectoryNotFoundException("CLIP tokenizer recipe missing.");
            string tokenizerDestination = Path.Combine(destination, "tokenizer");
            Directory.CreateDirectory(tokenizerDestination);
            foreach (string file in Directory.EnumerateFiles(tokenizer))
                File.Copy(file, Path.Combine(tokenizerDestination, Path.GetFileName(file)));
            string[] unused = checkpoint.Tensors.Keys.Except(used).Order().ToArray();
            var manifest = new { schemaVersion = 1, name = Path.GetFileNameWithoutExtension(checkpointPath),
                architecture, pipeline = "sd15-lcm", checkpoint = Path.GetFullPath(checkpointPath), checkpointSha256 = checkpointHash,
                graphRecipeDirectory = Path.GetFullPath(templateDirectory), sourceWeightCount = checkpoint.Tensors.Count,
                usedSourceWeightCount = used.Count, unusedSourceWeights = unused, stages = reports,
                status = "converted-requires-execution-validation", createdUtc = DateTimeOffset.UtcNow };
            File.WriteAllText(Path.Combine(destination, "model.json"), JsonSerializer.Serialize(manifest, Options));
            return manifest;
        }
        catch (Exception error)
        {
            File.WriteAllText(Path.Combine(destination, "conversion.failed.json"), JsonSerializer.Serialize(new { checkpointPath, templateDirectory, architecture, error = error.Message }, Options));
            throw;
        }
    }
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };
    private static object Difference(byte[] actual, byte[] reference, int dtype)
    {
        int count = actual.Length / (dtype == 10 ? 2 : 4), changed = 0;
        double maximum = 0, total = 0;
        for (int i = 0; i < count; i++)
        {
            float a = dtype == 10 ? (float)MemoryMarshal.Cast<byte, Half>(actual)[i] : MemoryMarshal.Cast<byte, float>(actual)[i];
            float b = dtype == 10 ? (float)MemoryMarshal.Cast<byte, Half>(reference)[i] : MemoryMarshal.Cast<byte, float>(reference)[i];
            if (!float.IsFinite(a) || !float.IsFinite(b)) throw new InvalidDataException("Non-finite weight in checkpoint or graph recipe.");
            double difference = Math.Abs((double)a - b);
            if (difference != 0) changed++;
            maximum = Math.Max(maximum, difference); total += difference;
        }
        return new { changedElements = changed, elements = count, maximumAbsoluteError = maximum, meanAbsoluteError = total / count };
    }
    private static string HashFile(string path) { using var file = File.OpenRead(path); return System.Convert.ToHexString(SHA256.HashData(file)); }
    private static void Metadata(Stream output, string key, string value)
    {
        using var entry = new MemoryStream(); OnnxWire.Text(entry, 1, key); OnnxWire.Text(entry, 2, value); OnnxWire.Bytes(output, 14, entry.ToArray());
    }
    private static void External(Stream output, string key, string value)
    {
        using var entry = new MemoryStream(); OnnxWire.Text(entry, 1, key); OnnxWire.Text(entry, 2, value); OnnxWire.Bytes(output, 13, entry.ToArray());
    }
    private static byte[] Transform(SafeTensorCheckpoint checkpoint, OnnxTensor target, WeightBinding binding)
    {
        int elementBytes = target.DataType == 10 ? 2 : 4;
        long count = 1; foreach (long dim in target.Shape) count = checked(count * dim);
        byte[] result = new byte[checked((int)(count * elementBytes))];
        if (binding.Transform == "packed-head-projections")
        {
            int parts = binding.Sources.Length, input = checked((int)target.Shape[0]), columns = checked((int)target.Shape[1]);
            if (target.Shape.Length != 2 || columns % (parts * binding.Heads) != 0) throw new InvalidDataException("Invalid packed projection dimensions.");
            int headWidth = columns / (parts * binding.Heads), output = columns / parts;
            for (int part = 0; part < parts; part++)
            {
                var source = checkpoint.Tensors[binding.Sources[part]];
                if (!source.Shape.SequenceEqual(new long[] { output, input })) throw new InvalidDataException($"Packed projection shape mismatch: {source.Name}.");
                byte[] bytes = checkpoint.Read(source);
                for (int head = 0; head < binding.Heads; head++)
                    for (int lane = 0; lane < headWidth; lane++)
                        for (int row = 0; row < input; row++)
                        {
                            int sourceIndex = (head * headWidth + lane) * input + row;
                            int targetIndex = row * columns + head * parts * headWidth + part * headWidth + lane;
                            Write(result, targetIndex, target.DataType, Read(bytes, sourceIndex, source.DataType));
                        }
            }
            return result;
        }
        var tensor = checkpoint.Tensors[binding.Sources.Single()];
        byte[] sourceBytes = checkpoint.Read(tensor);
        long sourceCount = 1; foreach (long dim in tensor.Shape) sourceCount = checked(sourceCount * dim);
        if (sourceCount != count) throw new InvalidDataException($"Element count mismatch: {tensor.Name} -> {target.Name}.");
        if (binding.Transform == "transpose")
        {
            if (tensor.Shape.Length < 2 || tensor.Shape.Skip(2).Any(d => d != 1) || target.Shape.Length != 2 ||
                target.Shape[0] != tensor.Shape[1] || target.Shape[1] != tensor.Shape[0])
                throw new InvalidDataException($"Transpose shape mismatch: {tensor.Name} -> {target.Name}.");
            int rows = checked((int)tensor.Shape[0]), cols = checked((int)tensor.Shape[1]);
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < cols; col++)
                    Write(result, col * rows + row, target.DataType, Read(sourceBytes, row * cols + col, tensor.DataType));
        }
        else
        {
            static long[] Squeezed(long[] shape) { int n = shape.Length; while (n > 1 && shape[n - 1] == 1) n--; return shape.Take(n).ToArray(); }
            if (!Squeezed(tensor.Shape).SequenceEqual(Squeezed(target.Shape)))
                throw new InvalidDataException($"Identity/reshape shape mismatch: {tensor.Name} -> {target.Name}.");
            if ((target.DataType == 10 && tensor.DataType == "F16") || (target.DataType == 1 && tensor.DataType == "F32"))
                return sourceBytes;
            for (int i = 0; i < count; i++) Write(result, i, target.DataType, Read(sourceBytes, i, tensor.DataType));
        }
        return result;
    }
    private static float Read(byte[] bytes, int index, string dtype) => dtype switch
    {
        "F16" => (float)MemoryMarshal.Cast<byte, Half>(bytes)[index],
        "F32" => MemoryMarshal.Cast<byte, float>(bytes)[index],
        _ => throw new NotSupportedException($"Weight conversion dtype {dtype}.")
    };
    private static void Write(byte[] bytes, int index, int dtype, float value)
    {
        if (dtype == 10) MemoryMarshal.Cast<byte, Half>(bytes)[index] = (Half)value;
        else MemoryMarshal.Cast<byte, float>(bytes)[index] = value;
    }
}
