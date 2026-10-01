using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DirectAI;

public class GenerativeEraseEngine : IDisposable
{
    private readonly DirectMLSession _session;

    public GenerativeEraseEngine(string modelPath, int deviceId)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"MI-GAN model not found at {modelPath}");
        }
        _session = new DirectMLSession(modelPath, deviceId);
    }

    public Image<Rgba32> Erase(Image<Rgba32> image, Image<L8> mask)
    {
        int w = image.Width;
        int h = image.Height;

        // Ensure dimensions are multiples of 8 for CNN strides
        int padW = ((w + 7) / 8) * 8;
        int padH = ((h + 7) / 8) * 8;

        var imgTensor = new DenseTensor<byte>(new[] { 1, 3, padH, padW });
        var maskTensor = new DenseTensor<byte>(new[] { 1, 1, padH, padW });

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var px = image[x, y];
                imgTensor[0, 0, y, x] = px.R;
                imgTensor[0, 1, y, x] = px.G;
                imgTensor[0, 2, y, x] = px.B;

                maskTensor[0, 0, y, x] = mask[x, y].PackedValue;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", imgTensor),
            NamedOnnxValue.CreateFromTensor("mask", maskTensor)
        };

        using var outputs = _session.Run(inputs);
        var resultTensor = outputs.First(o => o.Name == "result").AsTensor<byte>();

        var outImage = new Image<Rgba32>(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                byte r = resultTensor[0, 0, y, x];
                byte g = resultTensor[0, 1, y, x];
                byte b = resultTensor[0, 2, y, x];
                outImage[x, y] = new Rgba32(r, g, b, image[x, y].A);
            }
        }

        return outImage;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
