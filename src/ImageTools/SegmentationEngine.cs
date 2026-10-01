using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DirectAI;

public class SegmentationEngine : IDisposable
{
    private readonly DirectMLSession _encoderSession;
    private readonly DirectMLSession _decoderSession;

    // Cached embeddings per image to allow instant re-prompting / multi-click refinement
    private DenseTensor<float> _cachedEmbedding;
    private int _cachedWidth;
    private int _cachedHeight;

    public SegmentationEngine(string samModelDir, int deviceId)
    {
        string encPath = Path.Combine(samModelDir, "mobile_sam_image_encoder.onnx");
        string decPath = Path.Combine(samModelDir, "sam_mask_decoder_single.onnx");

        if (!File.Exists(encPath) || !File.Exists(decPath))
        {
            throw new FileNotFoundException($"MobileSAM models not found in {samModelDir}");
        }

        _encoderSession = new DirectMLSession(encPath, deviceId);
        _decoderSession = new DirectMLSession(decPath, deviceId);
    }

    public void EncodeImage(Image<Rgba32> image)
    {
        _cachedWidth = image.Width;
        _cachedHeight = image.Height;

        // Standard SAM pre-processing: resize to 1024x1024
        using var resized = image.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new Size(1024, 1024),
            Mode = ResizeMode.Stretch
        }));

        var inputTensor = new DenseTensor<float>(new[] { 1024, 1024, 3 });
        float[] mean = { 123.675f, 116.28f, 103.53f };
        float[] std = { 58.395f, 57.12f, 57.375f };

        for (int y = 0; y < 1024; y++)
        {
            for (int x = 0; x < 1024; x++)
            {
                var px = resized[x, y];
                inputTensor[y, x, 0] = (px.R - mean[0]) / std[0];
                inputTensor[y, x, 1] = (px.G - mean[1]) / std[1];
                inputTensor[y, x, 2] = (px.B - mean[2]) / std[2];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_image", inputTensor)
        };

        using var outputs = _encoderSession.Run(inputs);
        var emb = outputs.First(o => o.Name == "image_embeddings").AsTensor<float>();
        _cachedEmbedding = new DenseTensor<float>(emb.ToArray(), new[] { 1, 256, 64, 64 });
    }

    public Image<L8> SegmentPoints(IEnumerable<(float x, float y, float label)> points)
    {
        if (_cachedEmbedding == null)
        {
            throw new InvalidOperationException("No image currently encoded. Call EncodeImage first.");
        }

        var ptList = points.ToList();
        if (ptList.Count == 0)
        {
            return new Image<L8>(_cachedWidth, _cachedHeight);
        }

        int n = ptList.Count;
        var coordsTensor = new DenseTensor<float>(new[] { 1, n, 2 });
        var labelsTensor = new DenseTensor<float>(new[] { 1, n });

        float scaleX = 1024f / _cachedWidth;
        float scaleY = 1024f / _cachedHeight;

        for (int i = 0; i < n; i++)
        {
            coordsTensor[0, i, 0] = ptList[i].x * scaleX;
            coordsTensor[0, i, 1] = ptList[i].y * scaleY;
            labelsTensor[0, i] = ptList[i].label; // 1 = positive, 0 = negative
        }

        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMask = new DenseTensor<float>(new[] { 0.0f }, new[] { 1 });
        var origSize = new DenseTensor<float>(new[] { (float)_cachedHeight, (float)_cachedWidth }, new[] { 2 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings", _cachedEmbedding),
            NamedOnnxValue.CreateFromTensor("point_coords", coordsTensor),
            NamedOnnxValue.CreateFromTensor("point_labels", labelsTensor),
            NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMask),
            NamedOnnxValue.CreateFromTensor("orig_im_size", origSize)
        };

        using var outputs = _decoderSession.Run(inputs);
        var masksTensor = outputs.First(o => o.Name == "masks").AsTensor<float>();

        var maskImage = new Image<L8>(_cachedWidth, _cachedHeight);
        int outH = masksTensor.Dimensions[2];
        int outW = masksTensor.Dimensions[3];

        for (int y = 0; y < Math.Min(_cachedHeight, outH); y++)
        {
            for (int x = 0; x < Math.Min(_cachedWidth, outW); x++)
            {
                float logit = masksTensor[0, 0, y, x];
                maskImage[x, y] = logit > 0.0f ? new L8(255) : new L8(0);
            }
        }

        return maskImage;
    }

    /// <summary>
    /// Background removal: segments the subject and returns an RGBA cutout with transparent background.
    /// </summary>
    public Image<Rgba32> RemoveBackground(Image<Rgba32> sourceImage, IEnumerable<(float x, float y, float label)> foregroundPoints = null)
    {
        EncodeImage(sourceImage);

        // If no points provided, default to image center as the subject anchor
        var pts = foregroundPoints?.ToList() ?? new List<(float x, float y, float label)>
        {
            (sourceImage.Width / 2f, sourceImage.Height / 2f, 1f)
        };

        using var mask = SegmentPoints(pts);

        var cutout = new Image<Rgba32>(sourceImage.Width, sourceImage.Height);
        for (int y = 0; y < sourceImage.Height; y++)
        {
            for (int x = 0; x < sourceImage.Width; x++)
            {
                var srcPx = sourceImage[x, y];
                byte alpha = mask[x, y].PackedValue;
                // Soft edge / feathering
                cutout[x, y] = new Rgba32(srcPx.R, srcPx.G, srcPx.B, alpha);
            }
        }

        return cutout;
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}
