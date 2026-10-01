# DirectAI — Heterogeneous Multi-GPU / Multi-Die DirectML Inference Engine

## 1. Project Invariants & Purpose
DirectAI is a clean, owned, low-latency DirectML inference orchestrator for multi-die and heterogeneous multi-GPU configurations on Windows x64.
- **Zero Third-Party Wrapper Bloat**: No dependency on OnnxStack, sd.cpp, or fragile UI frameworks. Direct implementation on `Microsoft.ML.OnnxRuntime.DirectML`.
- **Heterogeneous & Multi-Die Support**: Designed generically for any multi-adapter topology:
  - Multi-die accelerators (e.g. dual-die AMD Radeon Pro V340L Vega 10 dies).
  - Cross-vendor heterogeneous configurations (e.g. NVIDIA Quadro P2000 alongside AMD Radeon Pro V340L dies).
  - Multi-card desktop and workstation rigs.
- **Permanent Stage Residency**: Sub-model stages remain resident on their assigned devices. Zero pipeline reloads between Text-to-Image, Image-to-Image, or Inpaint workflows.
- **Rapid Packed Math (FP16)**: DirectML compiles HLSL compute shaders natively utilizing half-precision math (such as Vega 10 Rapid Packed Math).

## 2. Hardware Topology on Host Machine
- **Device 0**: NVIDIA Quadro P2000 (PCIe, 5 GB GDDR5) — available for auxiliary/heterogeneous staging.
- **Device 1**: AMD Radeon Pro V340L Die 0 (PCIe, 8 GB HBM2) — Stage 0 (Text Encoder).
- **Device 2**: AMD Radeon Pro V340L Die 1 (PCIe, 8 GB HBM2) — Stage 1 (UNet Denoise Engine).
- **Device 3**: AMD Radeon Pro V340L Die 2 (PCIe, 8 GB HBM2) — Stage 2 (VAE Decoder).
- **Device 4**: AMD Radeon Pro V340L Die 3 (PCIe, 8 GB HBM2) — Stage 3 (ControlNet / Second Pass).
- **Device 5**: AMD Radeon Pro V340L Die 4 (PCIe, 8 GB HBM2) — Spare compute die.

## 3. Data & Tensor Contracts
- **CLIP Text Encoder**: Text Tokens $\to$ `prompt_embeds` (`[2, 77, 768]` ~600 KB - 1.2 MB).
- **UNet Denoise Engine**: `latents` (`[1, 4, 64, 64]` ~64 KB) + `timestep` + `prompt_embeds` $\to$ noise residuals.
- **VAE Decoder**: Final Latents (`[1, 4, 64, 64]`) $\to$ RGB Image (`[1, 3, 512, 512]` ~3 MB).
- **Boundary Transfers**: Minimal payload sizes crossing PCIe boundaries via DirectML/D3D12.

## 4. Environment Invariants
- Runtime: .NET 10 / .NET 11 (Windows x64).
- Redirect BitLocker-constrained variables (`DOTNET_CLI_HOME`, `NUGET_PACKAGES`, `NUGET_HTTP_CACHE_PATH`) to local non-locked storage (`C:\dev\DirectAI\.nuget`, etc.).
