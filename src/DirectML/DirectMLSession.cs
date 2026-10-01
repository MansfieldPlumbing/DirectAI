using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace DirectAI;

public class DirectMLSession : IDisposable
{
    private readonly string _modelPath;
    private readonly int _deviceId;
    private InferenceSession _session;

    public int DeviceId => _deviceId;
    public string ModelPath => _modelPath;
    public double RunMilliseconds { get; private set; }
    public int RunCount { get; private set; }
    public void ResetTimings() { RunMilliseconds = 0; RunCount = 0; }
    public IReadOnlyDictionary<string, NodeMetadata> InputMetadata => _session.InputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata => _session.OutputMetadata;

    public DirectMLSession(string modelPath, int deviceId)
    {
        _modelPath = modelPath;
        _deviceId = deviceId;
        _session = CreateSession(modelPath, deviceId);
    }

    private static InferenceSession CreateSession(string modelPath, int deviceId)
    {
        using var options = new SessionOptions();
        options.EnableMemoryPattern = false;
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

        // Bind directly to the specified DirectML device ID
        options.AppendExecutionProvider_DML(deviceId);

        return new InferenceSession(modelPath, options);
    }

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        try { return _session.Run(inputs); }
        finally { timer.Stop(); RunMilliseconds += timer.Elapsed.TotalMilliseconds; RunCount++; }
    }

    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
    }
}
