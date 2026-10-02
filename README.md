# DirectAI
A Windows ONNX Runtime/DirectML inference runtime with separate plugin, HTTP, named-pipe and CLI assemblies. PowerShell can load the runtime directly without launching the executable.

This is an independent project, not an Amuse fork. Existing model-package directory names describe installed assets; no model weights are distributed here.

## Status
SD1.5 LCM inference and placement of whole text-encoder/UNet/VAE stages across V340L dies have benchmark receipts. **A single UNet or language-model graph split across GPUs is not implemented.** Existing ORT session calls and managed tensor handoffs are not the proposed zero-host-wait graph executor. Arbitrary vendor/device support and SDXL conversion are not qualified.

The current build targets Windows x64 and .NET 11 preview. The host entry point is one line; application capabilities reside in DLLs. Build with:

```powershell
& C:\bin\pwsh\pwsh.exe -NoProfile -File .\Build-DirectAI.ps1
.\app\DirectAI.Host.exe capabilities --root $PWD
.\app\DirectAI.Host.exe invoke scripts.echo --root $PWD --request '{"hello":"world"}'
```

Or drive it directly:

```powershell
Import-Module .\powershell\DirectAI.psm1
Get-DirectAICapability
$runtime = New-DirectAIRuntime
try { Invoke-DirectAI $runtime scripts.echo @{ hello = 'world' } }
finally { Remove-DirectAIRuntime $runtime }
```

Plugin discovery reads `plugin.json` files without creating GPU devices or loading models. Assembly plugins also embed their manifest as `DirectAI.plugin.json`; their .NET dependency metadata describes assembly dependencies. The closure is not a hermetic model bundle: native runtime requirements and external model assets remain explicit dependencies. PS1 plugins accept JSON on stdin and return JSON on stdout.

## Layout
`projects/` contains independent Core, DirectML, Diffusion, ModelConversion, HTTP, NamedPipes, CLI, Routing, Host and image-tool project definitions. Core defines the plugin contract; transports invoke the same request contract. Plugins load on first invocation.

`powershell/` provides direct runtime access and [SD1.5 checkpoint conversion](powershell/MODEL-CONVERSION.md). Conversion rebinds a compatible ONNX template; it is not a general architecture exporter. `samples/` contains request examples. Generated artifacts go under `output/`, with JSON beside each image. `app/`, model weights and caches are ignored.

## Recorded performance
CyberRealistic-LCM family, 512×512, six LCM steps, seed 42, FP16, resident sessions, one warmup plus three measured images, AMD Pro 22Q4:

| TE / UNet / VAE adapter | DirectAI generation | sd.cpp Vulkan generation |
|---|---:|---:|
| 0 / 0 / 0 | 1.838 s | 4.900 s |
| 1 / 1 / 1 | 1.681 s | 4.810 s |
| 0 / 1 / 1 | 1.692 s | 4.830 s |
| 0 / 1 / 2 | 1.705 s | 4.810 s |

These are historical measurements from the pre-refactor Release executable. The original installed ONNX model and checkpoint-derived GGUF were **not fully weight-identical**. Each UNet stayed on one GPU; whole-stage placement is not UNet partitioning. Timings exclude loading and image encoding and have different runtime timer boundaries. The result is a comparable workload observation, not proof that DirectML is universally 2.8× faster. Subsequent exact checkpoint rebinding requires a new controlled head-to-head benchmark.

[Original measurements and weight audit](benchmarks/sd15-cyberrealistic/). The V340L experiment is documented in [V340L-Enablement](https://github.com/MansfieldPlumbing/V340L-Enablement).

## Next work
The [graph-builder implementation plan](docs/GRAPH-BUILDER-PLAN.md) starts with one whole ONNX graph, identifies legal seams, and compiles resource and GPU dependency contracts before execution. It is a plan, not a claim of current support.

## Dependencies
See [third-party notices](THIRD-PARTY-NOTICES.md). Plugins and PS1 scripts execute with the caller's privileges; discovery is not a sandbox. The local HTTP adapter binds loopback. Models have independent licenses.
