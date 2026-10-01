# Publication hygiene review
Date: 2026-10-01. Scope: the source tree and its two existing Git commits before initial public publication.

## Checks and repairs
- Inventoried source, project references, manifests and tracked history.
- Common credential signatures (GitHub tokens, AWS access-key IDs, OpenAI-style keys and private-key headers) returned zero matches in the publication file set and existing commit patches. This is a limited signature check, not a comprehensive security audit.
- No DLL/EXE, ONNX/GGUF/safetensors model, private key or certificate is in the publication file set. Local model paths and benchmark device identities remain as provenance.
- Ignored build/app output, caches, generated images and model weights. Original local output remains on disk.
- Removed the user-specific NuGet package path from repository configuration. The build script uses process-local C-drive cache locations.
- Added the missing transport/image-plugin project definitions and the ModelConversion aggregate reference.
- Kept Program.cs at one line. Archived unused duplicate host implementations outside compiled source.
- Embedded each assembly plugin manifest and retained directory-readable discovery manifests.
- Corrected plugin dependency packaging with CopyLocalLockFileAssemblies; initial device invocation exposed missing managed dependencies despite a successful compile.
- Restricted the named-pipe server to the current user. HTTP binds loopback. Scripts and plugins are trusted local code, not sandboxed.
- Included MIT source licensing and explicit ImageSharp split-license/dependency notices. External models are not redistributed.
- Preserved historical performance receipts and weight audits with qualifications.

## Verification
The current modular build completed with **0 warnings and 0 errors** using .NET SDK 11.0.100-preview.7.26381.103.
CLI discovery enumerated six plugins. CLI scripts.echo passed.
PowerShell loaded Core/Routing directly, invoked scripts.echo, and executed the resize plugin, producing an image plus adjacent JSON.
The Diffusion DLL exposed its embedded DirectAI.plugin.json resource. CLI device discovery passed after dependency packaging was fixed.
NuGet's vulnerable-package query for DirectAI.DirectML reported no vulnerable entries. This is time/source dependent and is not a guarantee that dependencies contain no vulnerabilities.

The existing SD1.5 timings came from the pre-refactor executable. This hygiene pass did not repeat that benchmark or qualify every inference plugin. No zero-host-wait multi-device graph execution is claimed.

## Current enumeration caveat
The enablement probe on this date enumerated five V340-labelled DXGI endpoints plus a P2000 and a software renderer. Adapter identity/physical topology must be resolved before any new benchmark; do not reinterpret historical four-die results using these current ordinals.

## Remaining scope
General ONNX architecture conversion, SDXL conversion and partitioned UNet/LLM inference are not implemented. See the graph-builder plan for gated follow-on work.
