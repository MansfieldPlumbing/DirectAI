using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DirectAI;

public class FaceSwapEngine : IDisposable
{
    private readonly DirectMLSession _arcFaceSession;
    private readonly DirectMLSession _swapperSession;
    private readonly int _deviceId;

    public int DeviceId => _deviceId;

    public FaceSwapEngine(string modelsDir, int deviceId = 0)
    {
        _deviceId = deviceId;
        string arcFacePath = Path.Combine(modelsDir, "w600k_r50.onnx");
        if (!File.Exists(arcFacePath))
        {
            arcFacePath = @"C:\dev\inswapper-restored\NextSwap\models\w600k_r50.onnx";
        }

        string swapperPath = Path.Combine(modelsDir, "inswapper_128.onnx");
        if (!File.Exists(swapperPath))
        {
            swapperPath = @"C:\dev\inswapper-restored\NextSwap\models\inswapper_128.onnx";
        }

        if (!File.Exists(arcFacePath))
            throw new FileNotFoundException($"ArcFace model not found at: {arcFacePath}");
        if (!File.Exists(swapperPath))
            throw new FileNotFoundException($"InSwapper model not found at: {swapperPath}");

        Console.WriteLine($"[FaceSwap] Initializing ArcFace on DirectML Device {deviceId}...");
        _arcFaceSession = new DirectMLSession(arcFacePath, deviceId);

        Console.WriteLine($"[FaceSwap] Initializing InSwapper on DirectML Device {deviceId}...");
        _swapperSession = new DirectMLSession(swapperPath, deviceId);
    }

    public float[] ExtractFaceEmbedding(Image<Rgba32> faceImage)
    {
        // ArcFace expects 1x3x112x112 normalized to [-1, 1]
        using var resized = faceImage.Clone(ctx => ctx.Resize(112, 112));
        var tensor = new float[3 * 112 * 112];

        for (int y = 0; y < 112; y++)
        {
            for (int x = 0; x < 112; x++)
            {
                var p = resized[x, y];
                tensor[0 * 112 * 112 + y * 112 + x] = (p.R / 127.5f) - 1.0f;
                tensor[1 * 112 * 112 + y * 112 + x] = (p.G / 127.5f) - 1.0f;
                tensor[2 * 112 * 112 + y * 112 + x] = (p.B / 127.5f) - 1.0f;
            }
        }

        var inputTensor = new DenseTensor<float>(tensor, new[] { 1, 3, 112, 112 });
        string inputName = _arcFaceSession.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        float[] rawEmbedding;
        using (var outputs = _arcFaceSession.Run(inputs))
        {
            var outVal = outputs.First();
            if (outVal.Value is DenseTensor<float> dt)
                rawEmbedding = dt.Buffer.ToArray();
            else
                rawEmbedding = outVal.AsEnumerable<float>().ToArray();
        }

        // L2 normalize embedding vector
        float norm = 0f;
        for (int i = 0; i < rawEmbedding.Length; i++)
        {
            norm += rawEmbedding[i] * rawEmbedding[i];
        }
        norm = MathF.Sqrt(norm);
        if (norm > 1e-6f)
        {
            for (int i = 0; i < rawEmbedding.Length; i++)
            {
                rawEmbedding[i] /= norm;
            }
        }

        return rawEmbedding;
    }

    public Image<Rgba32> SwapFace(Image<Rgba32> targetImage, float[] sourceEmbedding)
    {
        // InSwapper expects target face 1x3x128x128 normalized to [0, 1]
        using var target128 = targetImage.Clone(ctx => ctx.Resize(128, 128));
        var targetTensor = new float[3 * 128 * 128];

        for (int y = 0; y < 128; y++)
        {
            for (int x = 0; x < 128; x++)
            {
                var p = target128[x, y];
                targetTensor[0 * 128 * 128 + y * 128 + x] = p.R / 255.0f;
                targetTensor[1 * 128 * 128 + y * 128 + x] = p.G / 255.0f;
                targetTensor[2 * 128 * 128 + y * 128 + x] = p.B / 255.0f;
            }
        }

        var tTensor = new DenseTensor<float>(targetTensor, new[] { 1, 3, 128, 128 });
        var sTensor = new DenseTensor<float>(sourceEmbedding, new[] { 1, 512 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("target", tTensor),
            NamedOnnxValue.CreateFromTensor("source", sTensor)
        };

        float[] outPixels;
        using (var outputs = _swapperSession.Run(inputs))
        {
            var outVal = outputs.First();
            if (outVal.Value is DenseTensor<float> dt)
                outPixels = dt.Buffer.ToArray();
            else
                outPixels = outVal.AsEnumerable<float>().ToArray();
        }

        // Convert [1, 3, 128, 128] back to Image<Rgba32>
        var result = new Image<Rgba32>(128, 128);
        int plane = 128 * 128;
        for (int y = 0; y < 128; y++)
        {
            for (int x = 0; x < 128; x++)
            {
                int idx = y * 128 + x;
                byte r = (byte)Math.Clamp(MathF.Round(outPixels[0 * plane + idx] * 255.0f), 0, 255);
                byte g = (byte)Math.Clamp(MathF.Round(outPixels[1 * plane + idx] * 255.0f), 0, 255);
                byte b = (byte)Math.Clamp(MathF.Round(outPixels[2 * plane + idx] * 255.0f), 0, 255);
                result[x, y] = new Rgba32(r, g, b, 255);
            }
        }

        return result;
    }

    public Image<Rgba32> SwapFaceOntoCanvas(
        Image<Rgba32> canvas,
        Rectangle faceBounds,
        float[] sourceEmbedding,
        int featherRadius = 8)
    {
        using var faceCrop = canvas.Clone(ctx => ctx.Crop(faceBounds));
        using var swapped128 = SwapFace(faceCrop, sourceEmbedding);
        using var swappedFace = swapped128.Clone(ctx => ctx.Resize(faceBounds.Width, faceBounds.Height));

        var output = canvas.Clone();
        int fw = faceBounds.Width;
        int fh = faceBounds.Height;

        for (int y = 0; y < fh; y++)
        {
            int dstY = faceBounds.Y + y;
            if (dstY < 0 || dstY >= canvas.Height) continue;

            for (int x = 0; x < fw; x++)
            {
                int dstX = faceBounds.X + x;
                if (dstX < 0 || dstX >= canvas.Width) continue;

                // Feather blend at edges
                float distEdgeX = Math.Min(x, fw - 1 - x);
                float distEdgeY = Math.Min(y, fh - 1 - y);
                float distEdge = Math.Min(distEdgeX, distEdgeY);

                float alpha = featherRadius > 0 ? Math.Clamp(distEdge / featherRadius, 0.0f, 1.0f) : 1.0f;

                var orig = canvas[dstX, dstY];
                var swap = swappedFace[x, y];

                byte r = (byte)Math.Clamp(orig.R * (1.0f - alpha) + swap.R * alpha, 0, 255);
                byte g = (byte)Math.Clamp(orig.G * (1.0f - alpha) + swap.G * alpha, 0, 255);
                byte b = (byte)Math.Clamp(orig.B * (1.0f - alpha) + swap.B * alpha, 0, 255);

                output[dstX, dstY] = new Rgba32(r, g, b, 255);
            }
        }

        return output;
    }

    public void Dispose()
    {
        _arcFaceSession?.Dispose();
        _swapperSession?.Dispose();
    }
}
