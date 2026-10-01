using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DirectAI;

public class DiffusionPipeline : IDisposable
{
    private readonly DirectMLSession _textEncoder;
    private readonly DirectMLSession _unet;
    private readonly DirectMLSession _vaeDecoder;
    private readonly DirectMLSession _vaeEncoder;
    private readonly ClipTokenizer _tokenizer;
    private readonly LcmScheduler _scheduler;

    public int TextEncoderDevice => _textEncoder.DeviceId;
    public int UnetDevice => _unet.DeviceId;
    public int VaeDecoderDevice => _vaeDecoder.DeviceId;
    public object StageTimings => new
    {
        textEncoder = new { device = TextEncoderDevice, milliseconds = _textEncoder.RunMilliseconds, calls = _textEncoder.RunCount },
        unet = new { device = UnetDevice, milliseconds = _unet.RunMilliseconds, calls = _unet.RunCount },
        vaeDecoder = new { device = VaeDecoderDevice, milliseconds = _vaeDecoder.RunMilliseconds, calls = _vaeDecoder.RunCount }
    };

    private const float VaeScaleFactor = 0.18215f;

    public DiffusionPipeline(
        string modelDirectory,
        int textEncoderDeviceId = 0,
        int unetDeviceId = 0,
        int vaeDecoderDeviceId = 0)
    {
        string tePath = Path.Combine(modelDirectory, "text_encoder", "model.onnx");
        string unetPath = Path.Combine(modelDirectory, "unet", "model.onnx");
        string vdPath = Path.Combine(modelDirectory, "vae_decoder", "model.onnx");
        string vePath = Path.Combine(modelDirectory, "vae_encoder", "model.onnx");
        string tokPath = Path.Combine(modelDirectory, "tokenizer");
        foreach (string required in new[] { tePath, unetPath, vdPath })
            if (!File.Exists(required)) throw new FileNotFoundException("Required diffusion stage is missing.", required);
        string templatePath = Path.Combine(modelDirectory, "amuse_template.json");
        if (File.Exists(templatePath))
        {
            using var template = System.Text.Json.JsonDocument.Parse(File.ReadAllText(templatePath));
            if (template.RootElement.TryGetProperty("StableDiffusionTemplate", out var settings) &&
                settings.TryGetProperty("PipelineType", out var kind) && kind.GetString() != "LatentConsistency")
                throw new NotSupportedException("This pipeline currently implements SD 1.5 LCM. Base SD, SDXL, Cascade and SD3 require their own scheduler/tensor paths.");
        }

        Console.WriteLine($"[Pipeline] Initializing Stage 0: TextEncoder on DirectML Device {textEncoderDeviceId}...");
        _textEncoder = new DirectMLSession(tePath, textEncoderDeviceId);

        Console.WriteLine($"[Pipeline] Initializing Stage 1: UNet on DirectML Device {unetDeviceId}...");
        _unet = new DirectMLSession(unetPath, unetDeviceId);

        Console.WriteLine($"[Pipeline] Initializing Stage 2: VaeDecoder on DirectML Device {vaeDecoderDeviceId}...");
        _vaeDecoder = new DirectMLSession(vdPath, vaeDecoderDeviceId);

        if (File.Exists(vePath))
        {
            Console.WriteLine($"[Pipeline] Initializing VaeEncoder on DirectML Device {vaeDecoderDeviceId}...");
            _vaeEncoder = new DirectMLSession(vePath, vaeDecoderDeviceId);
        }

        Console.WriteLine($"[Pipeline] Loading CLIP Tokenizer from {tokPath}...");
        _tokenizer = new ClipTokenizer(tokPath);

        _scheduler = new LcmScheduler();
        Console.WriteLine($"[Pipeline] Multi-Device Pipeline Ready!");
    }

    public async Task<Image<Rgba32>> GenerateTextToImageAsync(
        string prompt,
        int steps = 6,
        int seed = 42,
        int width = 512,
        int height = 512,
        Action<int, int> onStep = null,
        CancellationToken ct = default)
    {
        return await Task.Run(() =>
        {
            if (steps < 1 || steps > 50 || width < 8 || height < 8 || width % 8 != 0 || height % 8 != 0)
                throw new ArgumentOutOfRangeException(nameof(steps), "Use 1–50 steps and positive dimensions divisible by eight.");
            _textEncoder.ResetTimings(); _unet.ResetTimings(); _vaeDecoder.ResetTimings();
            // 1. Tokenize prompt & negative prompt
            var tokens = _tokenizer.Tokenize(prompt);

            // 2. Stage 0: Encode prompt on TextEncoder device
            var teInputs = new List<NamedOnnxValue>
            {
                _textEncoder.InputMetadata["input_ids"].ElementType == typeof(long)
                    ? NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(tokens.Select(t => (long)t).ToArray(), new[] { 1, tokens.Length }))
                    : NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<int>(tokens, new[] { 1, tokens.Length }))
            };

            float[] promptEmbeds;
            using (var teOutputs = _textEncoder.Run(teInputs))
            {
                var hiddenState = teOutputs.First(o => o.Name == "last_hidden_state");
                promptEmbeds = ExtractFloatArray(hiddenState);
            }

            ct.ThrowIfCancellationRequested();

            // 3. Stage 1: Latent Consistency Denoising on UNet device
            int latentW = width / 8;
            int latentH = height / 8;
            var latents = _scheduler.GenerateInitialLatents(1, 4, latentH, latentW, seed);
            var timesteps = _scheduler.GenerateTimesteps(steps);
            var rng = seed >= 0 ? new Random(seed + 1) : new Random();

            for (int i = 0; i < steps; i++)
            {
                ct.ThrowIfCancellationRequested();
                int t = timesteps[i];
                int nextT = i < steps - 1 ? timesteps[i + 1] : 0;

                var unetInputs = new List<NamedOnnxValue>
                {
                    CreateTensor("sample", latents, new[] { 1, 4, latentH, latentW }, _unet.InputMetadata),
                    CreateTimestepTensor("timestep", t, _unet.InputMetadata),
                    CreateTensor("encoder_hidden_states", promptEmbeds, new[] { 1, 77, 768 }, _unet.InputMetadata)
                };

                float[] modelOutput;
                using (var unetOutputs = _unet.Run(unetInputs))
                {
                    modelOutput = ExtractFloatArray(unetOutputs.First());
                }

                // Denoising step
                latents = _scheduler.Step(modelOutput, t, latents, nextT, isLastStep: i == steps - 1, rng);
                onStep?.Invoke(i + 1, steps);
            }

            ct.ThrowIfCancellationRequested();

            // 4. Stage 2: Decode Latents on VAE Decoder device
            // Scale latents by 1 / 0.18215
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] /= VaeScaleFactor;
            }

            var vaeInputs = new List<NamedOnnxValue>
            {
                CreateTensor("latent_sample", latents, new[] { 1, 4, latentH, latentW }, _vaeDecoder.InputMetadata)
            };

            float[] rgbTensor;
            using (var vaeOutputs = _vaeDecoder.Run(vaeInputs))
            {
                rgbTensor = ExtractFloatArray(vaeOutputs.First());
            }

            // 5. Convert NCHW float [-1, 1] tensor to ImageSharp Image
            return ConvertTensorToImage(rgbTensor, width, height);
        }, ct);
    }

    public async Task<Image<Rgba32>> GenerateInpaintAsync(
        Image<Rgba32> initImage,
        Image<L8> maskImage,
        string prompt,
        float strength = 1.0f,
        int steps = 6,
        int seed = 42,
        Action<int, int> onStep = null,
        CancellationToken ct = default)
    {
        if (_vaeEncoder == null)
            throw new InvalidOperationException("VaeEncoder model not found for inpaint operation.");

        return await Task.Run(() =>
        {
            int width = initImage.Width;
            int height = initImage.Height;
            int latentW = width / 8;
            int latentH = height / 8;

            // 1. Encode init image to latents using VAE Encoder
            float[] initRgbTensor = ConvertImageToTensor(initImage);
            var initTensor = new DenseTensor<float>(initRgbTensor, new[] { 1, 3, height, width });
            var veInputs = new List<NamedOnnxValue>
            {
                CreateTensor("sample", initRgbTensor, new[] { 1, 3, height, width }, _vaeEncoder.InputMetadata)
            };

            float[] initLatents;
            using (var veOutputs = _vaeEncoder.Run(veInputs))
            {
                initLatents = ExtractFloatArray(veOutputs.First());
            }

            for (int i = 0; i < initLatents.Length; i++)
            {
                initLatents[i] *= VaeScaleFactor;
            }

            // Downscale mask to latent resolution [latentH, latentW]
            float[] latentMask = new float[latentH * latentW];
            for (int y = 0; y < latentH; y++)
            {
                for (int x = 0; x < latentW; x++)
                {
                    int srcX = Math.Clamp(x * 8 + 4, 0, width - 1);
                    int srcY = Math.Clamp(y * 8 + 4, 0, height - 1);
                    latentMask[y * latentW + x] = maskImage[srcX, srcY].PackedValue > 127 ? 1.0f : 0.0f;
                }
            }

            // 2. Encode prompt
            var tokens = _tokenizer.Tokenize(prompt);
            var teInputs = new List<NamedOnnxValue>
            {
                _textEncoder.InputMetadata["input_ids"].ElementType == typeof(long)
                    ? NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(tokens.Select(t => (long)t).ToArray(), new[] { 1, tokens.Length }))
                    : NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<int>(tokens, new[] { 1, tokens.Length }))
            };

            float[] promptEmbeds;
            using (var teOutputs = _textEncoder.Run(teInputs))
            {
                promptEmbeds = ExtractFloatArray(teOutputs.First(o => o.Name == "last_hidden_state"));
            }

            // 3. Setup initial latents
            var noiseLatents = _scheduler.GenerateInitialLatents(1, 4, latentH, latentW, seed);
            var latents = new float[noiseLatents.Length];
            for (int c = 0; c < 4; c++)
            {
                for (int y = 0; y < latentH; y++)
                {
                    for (int x = 0; x < latentW; x++)
                    {
                        int idx = c * latentH * latentW + y * latentW + x;
                        float m = latentMask[y * latentW + x];
                        latents[idx] = m > 0.5f ? noiseLatents[idx] : initLatents[idx];
                    }
                }
            }

            var timesteps = _scheduler.GenerateTimesteps(steps);
            var rng = seed >= 0 ? new Random(seed + 1) : new Random();

            for (int i = 0; i < steps; i++)
            {
                ct.ThrowIfCancellationRequested();
                int t = timesteps[i];
                int nextT = i < steps - 1 ? timesteps[i + 1] : 0;

                var unetInputs = new List<NamedOnnxValue>
                {
                    CreateTensor("sample", latents, new[] { 1, 4, latentH, latentW }, _unet.InputMetadata),
                    CreateTimestepTensor("timestep", t, _unet.InputMetadata),
                    CreateTensor("encoder_hidden_states", promptEmbeds, new[] { 1, 77, 768 }, _unet.InputMetadata)
                };

                float[] modelOutput;
                using (var unetOutputs = _unet.Run(unetInputs))
                {
                    modelOutput = ExtractFloatArray(unetOutputs.First());
                }

                var stepLatents = _scheduler.Step(modelOutput, t, latents, nextT, isLastStep: i == steps - 1, rng);

                // Inpainting blending: keep unmasked areas faithful to initLatents
                for (int c = 0; c < 4; c++)
                {
                    for (int y = 0; y < latentH; y++)
                    {
                        for (int x = 0; x < latentW; x++)
                        {
                            int idx = c * latentH * latentW + y * latentW + x;
                            float m = latentMask[y * latentW + x];
                            latents[idx] = m > 0.5f ? stepLatents[idx] : initLatents[idx];
                        }
                    }
                }

                onStep?.Invoke(i + 1, steps);
            }

            // 4. Decode
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] /= VaeScaleFactor;
            }

            var vaeInputs = new List<NamedOnnxValue>
            {
                CreateTensor("latent_sample", latents, new[] { 1, 4, latentH, latentW }, _vaeDecoder.InputMetadata)
            };

            float[] rgbTensor;
            using (var vaeOutputs = _vaeDecoder.Run(vaeInputs))
            {
                rgbTensor = ExtractFloatArray(vaeOutputs.First());
            }

            return ConvertTensorToImage(rgbTensor, width, height);
        }, ct);
    }

    private static float[] ConvertImageToTensor(Image<Rgba32> image)
    {
        int w = image.Width;
        int h = image.Height;
        var tensor = new float[3 * h * w];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var pixel = image[x, y];
                tensor[0 * h * w + y * w + x] = (pixel.R / 127.5f) - 1.0f;
                tensor[1 * h * w + y * w + x] = (pixel.G / 127.5f) - 1.0f;
                tensor[2 * h * w + y * w + x] = (pixel.B / 127.5f) - 1.0f;
            }
        }

        return tensor;
    }

    private static Image<Rgba32> ConvertTensorToImage(float[] rgbTensor, int width, int height)
    {
        var image = new Image<Rgba32>(width, height);
        int channelSize = width * height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                float r = (rgbTensor[0 * channelSize + idx] / 2.0f + 0.5f) * 255.0f;
                float g = (rgbTensor[1 * channelSize + idx] / 2.0f + 0.5f) * 255.0f;
                float b = (rgbTensor[2 * channelSize + idx] / 2.0f + 0.5f) * 255.0f;

                byte cr = (byte)Math.Clamp(MathF.Round(r), 0, 255);
                byte cg = (byte)Math.Clamp(MathF.Round(g), 0, 255);
                byte cb = (byte)Math.Clamp(MathF.Round(b), 0, 255);

                image[x, y] = new Rgba32(cr, cg, cb, 255);
            }
        }

        return image;
    }

    private static float[] ExtractFloatArray(DisposableNamedOnnxValue value)
    {
        if (value.Value is DenseTensor<float> denseFloat)
        {
            return denseFloat.Buffer.ToArray();
        }
        if (value.Value is DenseTensor<Float16> denseF16)
        {
            var span = denseF16.Buffer.Span;
            var result = new float[span.Length];
            for (int i = 0; i < span.Length; i++)
            {
                result[i] = span[i].ToFloat();
            }
            return result;
        }
        if (value.AsTensor<float>() is DenseTensor<float> dt)
        {
            return dt.Buffer.ToArray();
        }
        if (value.AsTensor<Float16>() is DenseTensor<Float16> dt16)
        {
            var span = dt16.Buffer.Span;
            var result = new float[span.Length];
            for (int i = 0; i < span.Length; i++)
            {
                result[i] = span[i].ToFloat();
            }
            return result;
        }
        var enumVal = value.AsEnumerable<float>();
        if (enumVal != null) return enumVal.ToArray();

        throw new NotSupportedException($"Cannot extract float array from {value.Name} of type {value.Value?.GetType().FullName}");
    }

    private static NamedOnnxValue CreateTensor(string name, float[] data, int[] dimensions, IReadOnlyDictionary<string, NodeMetadata> metadata)
    {
        if (metadata.TryGetValue(name, out var nodeMeta) && nodeMeta.ElementType == typeof(Float16))
        {
            var f16Data = new Float16[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                f16Data[i] = (Float16)data[i];
            }
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(f16Data, dimensions));
        }
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, dimensions));
    }

    private static NamedOnnxValue CreateTimestepTensor(string name, float timestep, IReadOnlyDictionary<string, NodeMetadata> metadata)
    {
        if (metadata.TryGetValue(name, out var nodeMeta))
        {
            if (nodeMeta.ElementType == typeof(long) || nodeMeta.ElementType == typeof(Int64))
            {
                return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(new[] { (long)timestep }, new[] { 1 }));
            }
            if (nodeMeta.ElementType == typeof(int) || nodeMeta.ElementType == typeof(Int32))
            {
                return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(new[] { (int)timestep }, new[] { 1 }));
            }
            if (nodeMeta.ElementType == typeof(Float16))
            {
                return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(new[] { (Float16)timestep }, new[] { 1 }));
            }
        }
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new[] { timestep }, new[] { 1 }));
    }

    public void Dispose()
    {
        _textEncoder?.Dispose();
        _unet?.Dispose();
        _vaeDecoder?.Dispose();
        _vaeEncoder?.Dispose();
    }
}
