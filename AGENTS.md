# DirectAI — DirectML ONNX Inferencing Engine for Windows

## 1. Project Invariants & Purpose
DirectAI is a lightweight, low-latency, modular inference engine for ONNX models running on Windows via DirectML.
- **Architectural Scope**: Supports both single-GPU and multi-GPU / multi-die topologies generically.
- **Cross-Vendor Heterogeneity**: Works across any DirectX 12 compatible hardware (AMD, NVIDIA, Intel, Qualcomm Snapdragon X) in homogeneous, multi-die, or heterogeneous configurations.
- **Direct Runtime**: Built directly on `Microsoft.ML.OnnxRuntime.DirectML`. Zero dependency on third-party frameworks or wrapper bloat.
- **Permanent Stage Residency**: Sub-model pipeline stages remain resident in VRAM on their assigned hardware device, eliminating thrashing and reloads between inferencing workflows.
- **Minimal Host Boundary Overhead**: Inter-stage activations (embeddings, latent representations, decoded outputs) are transferred with minimal host/memory overhead between execution devices.

## 2. Multi-Device Orchestration Model
DirectAI abstracts hardware adapters into logical compute devices:
- **Device Discovery**: Dynamically queries DXGI/DirectML adapters at startup. Discovers vendor, dedicated VRAM, shared memory, and hardware flags.
- **Flexible Device Mapping**:
  - **Single-GPU Mode**: Executes all pipeline stages on a single selected adapter.
  - **Multi-GPU / Multi-Die Pipeline Mode**: Distributes modular stages (e.g., Text Encoder, UNet / DiT denoiser, ControlNet, VAE Decoder) across distinct adapters.
  - **Heterogeneous Mode**: Distributes stages across GPUs from different vendors (e.g. NVIDIA + AMD) or asymmetric VRAM tiers.
- **Automatic Budgeting**: Stages can be placed based on available dedicated VRAM and compute profile.

## 3. Data & Tensor Contracts
- **Modular Diffusion Pipelines**:
  - `TextEncoder`: Tokens $\to$ Text Embeddings (~1 MB).
  - `UNet / DiT`: Latent steps ($4 \times H/8 \times W/8$) $\to$ Predicted noise (~64 KB - 256 KB).
  - `VAE Decoder`: Final Latents $\to$ RGB Output (~3 MB).
- **Execution Lifecycle**:
  - Pipelines load once into assigned devices and remain active.
  - Workflows (Text-to-Image, Image-to-Image, Inpainting, ControlNet) share the resident UNet and VAE without eviction.

## 4. Environment & Platform
- OS: Windows x64.
- Runtime: .NET 10 / .NET 11.
- Target EP: `DirectMLExecutionProvider`.
