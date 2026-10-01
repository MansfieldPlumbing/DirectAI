# DirectAI repository instructions
DirectAI is an independent Windows ONNX Runtime/DirectML project. It is not an Amuse or TensorStack fork.

## Scope and evidence
Current demonstrated inference is SD1.5 LCM with whole text-encoder, UNet and VAE stages placed on selected adapters. Partitioned UNet/LLM graphs, general conversion, SDXL conversion and arbitrary hardware support are not demonstrated. Do not describe planned features as implemented.

Keep benchmark weights, shapes, precision, timer boundaries, warmup and physical adapter identities explicit. Historical ONNX/GGUF comparison weights were not fully identical. Do not claim a universal API speed ratio.

## Architecture
Keep Program.cs minimal. Core contracts, transports, routing, compute and plugins belong in separate projects. HTTP, named pipes and CLI use the same capability contract. PowerShell is a first-class direct entry point; the executable is optional.

Discovery reads plugin manifests from a directory without loading devices or models. Embed assembly plugin manifests. Declare native and external model requirements; do not imply a DLL contains a complete model closure.

Generated outputs go in output/, with JSON metadata beside each output. Do not publish models, caches, binaries, credentials or local outputs. Preserve third-party licenses.

## Graph work
Follow docs/GRAPH-BUILDER-PLAN.md. Begin with one whole graph, nominate legal seams and build complete live-value contracts before execution. Do not substitute a dynamic runtime scheduler/coordinator for graph analysis.

For the proposed multi-device graph hot path, CPU waits, polling, queue drains and managed memcpy/readback bridges across devices are prohibited. Native GPU dependencies enforce causality. Existing managed ORT stage execution is not claimed to satisfy that future contract. Target D3D12 feature level 12_1; do not assume 12_2 hardware.

## Operations
Use explicit executable paths. Build-DirectAI.ps1 sets process-local C-drive .NET/NuGet locations. Preserve user changes and raw benchmark evidence. Do not rename immutable proof paths in the separate V340L repository.
