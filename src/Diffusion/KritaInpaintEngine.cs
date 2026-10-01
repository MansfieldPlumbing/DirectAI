using System;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DirectAI;

public static class KritaInpaintEngine
{
    /// <summary>
    /// Executes selection-aware inpainting following Krita Diffusion's architecture:
    /// 1. Mask bounding box identification
    /// 2. Context expansion margin (preserving surrounding lighting and texture)
    /// 3. Square aspect ratio adjustment & multiple-of-8 alignment
    /// 4. Sub-canvas cropping & bicubic normalization to model resolution (512x512)
    /// 5. Differential latent diffusion
    /// 6. Resize back to context bounds
    /// 7. Feathered alpha edge compositing to eliminate seam boundaries
    /// </summary>
    public static async Task<Image<Rgba32>> InpaintAsync(
        DiffusionPipeline pipeline,
        Image<Rgba32> canvas,
        Image<L8> mask,
        string prompt,
        int steps = 6,
        int seed = 42,
        float contextPaddingRatio = 0.35f,
        int featherRadius = 12,
        int modelResolution = 512,
        Action<int, int> onStep = null,
        CancellationToken ct = default)
    {
        int canvasW = canvas.Width;
        int canvasH = canvas.Height;

        // 1. Calculate active mask bounding box
        int minX = int.MaxValue, minY = int.MaxValue;
        int maxX = int.MinValue, maxY = int.MinValue;
        bool hasMask = false;

        for (int y = 0; y < canvasH; y++)
        {
            for (int x = 0; x < canvasW; x++)
            {
                if (mask[x, y].PackedValue > 15)
                {
                    hasMask = true;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (!hasMask)
        {
            // No mask drawn, return unmodified canvas copy
            return canvas.Clone();
        }

        // 2. Compute context expansion margin
        int maskW = maxX - minX + 1;
        int maskH = maxY - minY + 1;
        int padX = Math.Max(32, (int)(maskW * contextPaddingRatio));
        int padY = Math.Max(32, (int)(maskH * contextPaddingRatio));

        int centerX = minX + maskW / 2;
        int centerY = minY + maskH / 2;

        int targetSide = Math.Max(maskW + 2 * padX, maskH + 2 * padY);
        // Snap to multiple of 8
        targetSide = ((targetSide + 7) / 8) * 8;
        targetSide = Math.Clamp(targetSide, 128, Math.Max(canvasW, canvasH));

        int cropX = Math.Clamp(centerX - targetSide / 2, 0, Math.Max(0, canvasW - targetSide));
        int cropY = Math.Clamp(centerY - targetSide / 2, 0, Math.Max(0, canvasH - targetSide));
        int cropW = Math.Min(targetSide, canvasW - cropX);
        int cropH = Math.Min(targetSide, canvasH - cropY);

        // Snap crop dimensions to multiple of 8
        cropW = (cropW / 8) * 8;
        cropH = (cropH / 8) * 8;

        var cropRect = new Rectangle(cropX, cropY, cropW, cropH);

        // 3. Crop sub-image and sub-mask, scale to model resolution
        using var croppedCanvas = canvas.Clone(ctx => ctx.Crop(cropRect).Resize(modelResolution, modelResolution));
        using var croppedMask = mask.Clone(ctx => ctx.Crop(cropRect).Resize(modelResolution, modelResolution));

        // 4. Differential Latent Inpainting
        using var inpaintedPatch = await pipeline.GenerateInpaintAsync(
            initImage: croppedCanvas,
            maskImage: croppedMask,
            prompt: prompt,
            steps: steps,
            seed: seed,
            onStep: onStep,
            ct: ct);

        // 5. Resize inpainted patch back to crop bounds
        using var scaledPatch = inpaintedPatch.Clone(ctx => ctx.Resize(cropW, cropH));

        // 6. Generate Feathered Alpha Mask for smooth border blending
        var featheredMask = GenerateFeatheredAlphaMask(mask, cropRect, featherRadius);

        // 7. Composite seamlessly onto output canvas
        var result = canvas.Clone();
        for (int y = 0; y < cropH; y++)
        {
            int dstY = cropY + y;
            for (int x = 0; x < cropW; x++)
            {
                int dstX = cropX + x;
                float alpha = featheredMask[x, y];
                if (alpha <= 0.001f) continue;

                var orig = canvas[dstX, dstY];
                var patchPix = scaledPatch[x, y];

                byte r = (byte)Math.Clamp(orig.R * (1.0f - alpha) + patchPix.R * alpha, 0, 255);
                byte g = (byte)Math.Clamp(orig.G * (1.0f - alpha) + patchPix.G * alpha, 0, 255);
                byte b = (byte)Math.Clamp(orig.B * (1.0f - alpha) + patchPix.B * alpha, 0, 255);

                result[dstX, dstY] = new Rgba32(r, g, b, 255);
            }
        }

        return result;
    }

    private static float[,] GenerateFeatheredAlphaMask(Image<L8> fullMask, Rectangle cropRect, int radius)
    {
        int w = cropRect.Width;
        int h = cropRect.Height;
        var raw = new float[w, h];

        // Extract binary mask for the crop area
        for (int y = 0; y < h; y++)
        {
            int srcY = cropRect.Y + y;
            for (int x = 0; x < w; x++)
            {
                int srcX = cropRect.X + x;
                raw[x, y] = fullMask[srcX, srcY].PackedValue > 127 ? 1.0f : 0.0f;
            }
        }

        if (radius <= 0) return raw;

        // Two-pass separable box/Gaussian blur for smooth feathering
        var temp = new float[w, h];
        var blurred = new float[w, h];

        int kernelSize = radius * 2 + 1;
        float invKernel = 1.0f / kernelSize;

        // Horizontal pass
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                for (int k = -radius; k <= radius; k++)
                {
                    int kx = Math.Clamp(x + k, 0, w - 1);
                    sum += raw[kx, y];
                }
                temp[x, y] = sum * invKernel;
            }
        }

        // Vertical pass
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                for (int k = -radius; k <= radius; k++)
                {
                    int ky = Math.Clamp(y + k, 0, h - 1);
                    sum += temp[x, ky];
                }
                blurred[x, y] = sum * invKernel;
            }
        }

        return blurred;
    }
}
