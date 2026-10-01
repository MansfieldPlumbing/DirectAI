# Checkpoint conversion

The current admitted path converts an LDM-layout SD1.5 safetensors checkpoint into four ONNX stages using a compatible, existing SD1.5 graph recipe. It does not generate an arbitrary architecture from a bag of weights. Safetensors supplies tensors; the graph recipe supplies operators, inputs, outputs, and tokenizer assets.

Run from PowerShell (no Python required):

```powershell
& C:\bin\pwsh\pwsh.exe -NoProfile -File C:\dev\DirectAI\powershell\Convert-DirectAIModel.ps1 `
  -Build `
  -Checkpoint N:\models\checkpoints\cyberrealisticLCM_cyberrealistic42.safetensors `
  -TemplateDirectory C:\Models\DirectAI\CyberRealistic-LCM-amuse `
  -OutputDirectory C:\Models\DirectAI\CyberRealistic-LCM-DirectAI
```

The destination must be new or empty. The tested checkpoint is CyberRealistic-LCM 4.2. The current model manifest declares the SD1.5 LCM pipeline; other schedulers and architecture families have not been admitted.

The conversion DLL is independent of the executable, HTTP, and ORT. PowerShell loads it directly. Its project is `projects/DirectAI.ModelConversion.csproj`, with no external package dependencies.

Each stage contains `model.onnx`, external `model.onnx.data`, and `conversion.json`. The root contains `model.json` and copied tokenizer assets. Reports record source keys, transposes, packed attention projections, offsets, tensor hashes, checkpoint hash, graph recipe hash, and numerical differences from the recipe's original weights. Floating initializers are rebound from the checkpoint. Unknown source tensors fail conversion, except an explicit allowlist of schedule buffers, position IDs, and EMA bookkeeping.

The writer preserves non-initializer graph fields. It handles linear transposes, fused per-head QKV/KV packing, and normalization parameter mappings. Missing mappings, malformed safetensors extents, unsupported dtypes, and incompatible shapes fail. Failed writes leave `conversion.failed.json`; such directories must not be treated as executable models. Conversion does not claim execution validation automatically.

Validate emitted graph programs and weight payloads:

```powershell
& C:\bin\pwsh\pwsh.exe -NoProfile -File C:\dev\DirectAI\powershell\Test-DirectAIConversion.ps1
```

The receipt also checks rejection of existing output, malformed headers, and unsupported SDXL. Runtime evidence is stored separately beside generated images under `output/onnx-conversion-validation`. The current four stage graphs and weights have identical hashes to the artifacts used in that execution test.

## Verified receipt — 2026-10-01

- Independent converter build: zero warnings and errors.
- Source: 1,145 tensors; 1,130 used. Remaining 15 tensors are explicitly permitted bookkeeping buffers.
- Bound initializers: text encoder 196, UNet 638, VAE decoder 140, VAE encoder 108.
- All four ONNX sessions initialize with ORT DirectML. Text encoder, UNet, and VAE decoder execute a six-step 512×512 LCM image on device 1.
- Measured warm inference runs: 1,696.3749 ms and 1,681.6908 ms. Generated PNGs have identical hashes and depict coherent roses in a glass vase.
- VAE encoder initialization is verified; an image-encoding numerical test is still needed.
- Graph program and external payload integrity checks pass. Full numerical parity against an independent source-framework execution is not yet established. The installed graph recipe's text encoder weights differ from this source checkpoint; its images are not an exact-weight oracle.

## Remaining admission work

SDXL requires a separate dual-text-encoder recipe, conditioning contracts, checkpoint mappings, and runtime support. It is rejected explicitly today. The local RealVisXL checkpoint has not been converted. BF16 weight transforms and arbitrary graph families are also unsupported. Do not advertise this as a universal safetensors-to-ONNX exporter.

UNet partitioning remains a separate step after conversion correctness. No partitioning or multi-die speedup is established by this conversion receipt.
