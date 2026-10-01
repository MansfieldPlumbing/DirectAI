using System;
using System.Collections.Generic;

namespace DirectAI;

public class LcmScheduler
{
    private readonly int _trainSteps;
    private readonly float[] _betas;
    private readonly float[] _alphas;
    private readonly float[] _alphasCumprod;
    private readonly float[] _sigmas;

    public LcmScheduler(int trainSteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f)
    {
        _trainSteps = trainSteps;
        _betas = new float[trainSteps];
        _alphas = new float[trainSteps];
        _alphasCumprod = new float[trainSteps];
        _sigmas = new float[trainSteps];

        float start = MathF.Sqrt(betaStart);
        float end = MathF.Sqrt(betaEnd);
        float cumprod = 1.0f;

        for (int i = 0; i < trainSteps; i++)
        {
            float t = i / (float)(trainSteps - 1);
            float betaSqrt = start + t * (end - start);
            _betas[i] = betaSqrt * betaSqrt;
            _alphas[i] = 1.0f - _betas[i];
            cumprod *= _alphas[i];
            _alphasCumprod[i] = cumprod;
            _sigmas[i] = MathF.Sqrt((1.0f - cumprod) / cumprod);
        }
    }

    public int[] GenerateTimesteps(int numSteps, int originalSteps = 50)
    {
        int k = _trainSteps / originalSteps; // 1000 / 50 = 20
        var originTimesteps = new int[originalSteps];
        for (int i = 0; i < originalSteps; i++)
        {
            originTimesteps[i] = (i + 1) * k - 1; // 19, 39, ..., 999
        }
        Array.Reverse(originTimesteps); // [999, 979, ..., 19]

        var timesteps = new int[numSteps];
        for (int i = 0; i < numSteps; i++)
        {
            int idx = (int)MathF.Floor((float)i * originalSteps / numSteps);
            timesteps[i] = originTimesteps[idx];
        }
        return timesteps;
    }

    public float[] GenerateInitialLatents(int batchSize, int channels, int height, int width, int seed)
    {
        int length = batchSize * channels * height * width;
        var latents = new float[length];
        var rng = seed >= 0 ? new Random(seed) : new Random();

        // Standard Box-Muller Gaussian random noise
        for (int i = 0; i < length; i += 2)
        {
            double u1 = Math.Max(rng.NextDouble(), 1e-7);
            double u2 = rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;

            latents[i] = (float)(radius * Math.Cos(theta));
            if (i + 1 < length)
            {
                latents[i + 1] = (float)(radius * Math.Sin(theta));
            }
        }

        return latents;
    }

    public float[] Step(float[] modelOutput, int timestep, float[] sample, int nextTimestep, bool isLastStep, Random rng = null)
    {
        int length = sample.Length;
        var prevSample = new float[length];

        float alphaProdT = _alphasCumprod[timestep];
        float alphaProdTPrev = nextTimestep >= 0 ? _alphasCumprod[nextTimestep] : 1.0f;
        float betaProdT = 1.0f - alphaProdT;
        float betaProdTPrev = 1.0f - alphaProdTPrev;

        float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
        float sqrtBetaProdT = MathF.Sqrt(betaProdT);
        float sqrtAlphaProdTPrev = MathF.Sqrt(alphaProdTPrev);
        float sqrtBetaProdTPrev = MathF.Sqrt(betaProdTPrev);

        // HuggingFace Diffusers LCMScheduler boundary condition scalings
        const float sigmaData = 0.5f;
        const float timestepScaling = 10.0f;
        float scaledTimestep = timestep * timestepScaling;
        float scaledSq = scaledTimestep * scaledTimestep;
        float sigmaDataSq = sigmaData * sigmaData;

        float cSkip = sigmaDataSq / (scaledSq + sigmaDataSq);
        float cOut = scaledTimestep / MathF.Sqrt(scaledSq + sigmaDataSq);

        for (int i = 0; i < length; i++)
        {
            // UNet predicts epsilon (noise)
            float predictedX0 = (sample[i] - sqrtBetaProdT * modelOutput[i]) / sqrtAlphaProdT;

            // Denoise using consistency boundary conditions
            float denoised = cOut * predictedX0 + cSkip * sample[i];

            if (!isLastStep)
            {
                float noise = rng != null ? NextGaussian(rng) : 0f;
                prevSample[i] = sqrtAlphaProdTPrev * denoised + sqrtBetaProdTPrev * noise;
            }
            else
            {
                prevSample[i] = denoised;
            }
        }

        return prevSample;
    }

    private static float NextGaussian(Random rng)
    {
        double u1 = Math.Max(rng.NextDouble(), 1e-7);
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
