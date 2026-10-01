# Whole-graph seams and compiled GPU contracts
Implementation plan, 2026-10-01. No multi-device LLM executor is implemented by this document.

## Mechanism to preserve
Start with one complete ONNX graph. Find seams where the computation can be cut and rejoined without losing values or needlessly breaking useful fusion. If the export recipe hides those seams, change the recipe to expose equivalent boundaries. Do this analysis, measurement and contract construction **before execution**.

Runtime replays the compiled GPU commands and dependencies. It does not discover tasks, pick devices, decide when a stage is ready, poll completion, wake a CPU coordinator or reconstruct tensor semantics at every boundary. Device ordering remains necessary: GPU queues publish, wait, consume and release through native fence/resource operations. A predicted “beat” is a profiling observation, not permission to read unfinished data.

Target D3D12 feature level **12_1**. Do not require 12_2, Work Graphs, Vulkan, or a new universal kernel library. Keep existing ORT/DirectML execution where it satisfies the contract.

## Deliverable 1: inspect one real graph
Build a model-independent graph analysis DLL and PS1 front end first. Input: the exact model, external weight data, ONNX opsets, shape bounds, target adapter capabilities and intended prefill/decode shapes.

Produce a producer/consumer graph, control-flow subgraph inventory, topological regions, repeated transformer blocks, residual branches/joins, initializer ownership and inferred value types/shapes. Treat unknown shapes and control flow explicitly; do not estimate them as zero bytes. Preserve external-data references and hashes.

For each candidate region A, the boundary is every value produced in A and consumed outside A. A skip connection has the same rule: its tensor remains live until its last consumer, even if many blocks lie between. Gemma may also have per-layer inputs and persistent KV state; a seam is not assumed to contain just one hidden-state tensor.

Start with transformer block boundaries and legal post-residual joins. Use optimized graph inspection and selective profiling to see whether those boundaries preserve attention, quantization, dequantization and KV-update fusion. Never slice inside an opaque compiled fused operator. A source-level seam can be expensive if recompilation destroys fusion.

Output `analysis.json` with node/value IDs, candidate seam sets, live boundary tensors, byte sizes/ranges, live-state budgets and reasons a candidate is rejected. Discovery is file reading, not GPU initialization.

## Deliverable 2: nominate and measure a small set of seams
Profile the unsplit graph on one V340 die with fixed model/prompt/shape/precision. Use device timestamp ranges for candidate regions and separately record command recording/submission costs. Inspect three to five promising seam configurations, not every possible cut at runtime.

The cadence metaphor becomes measured region duration and branch arrival skew. A join is ready when all its input dependencies complete; no wall-clock synchronization or periodic timecodes are injected into the graph. In a branched graph, the boundary contract includes every live branch, including long-lived residuals.

Debug tracers may use diagnostic graph outputs or instrumented command markers. Record whether optimization eliminates or fuses a marker. A missing marker does not prove the tensor vanished or that the branch is unused. Remove diagnostic outputs from release performance runs because they can change fusion and lifetime.

Score measured candidates using:
- Decode critical-path latency: sum of dependent stage execution and transfer costs, including recompile/fusion penalties.
- VRAM fit: weights, local KV, temporaries and boundary lifetime overlap on each die.
- Boundary bytes, resource layout conversions and actual transfer cost.
- Throughput for multiple independent requests, only as a separate objective.

Identical dies simplify the first test; they do not make sequential transformer layers independent. Four GPUs primarily expand capacity. A single token still follows its causal layer chain. Capacity success with a small latency penalty is a useful result even without a speedup.

## Deliverable 3: emit partitions and a static contract
The builder emits ONNX partitions with complete boundary inputs/outputs, local initializers, persistent state ownership and deterministic naming. Separate compiled variants cover prefill and bounded decode shapes; select the appropriate variant at invocation, not at a device boundary.

Each cut contract contains:
- Value identity, dtype, shape bounds, byte layout/strides and resource size/alignment.
- Producing/consuming device identities, queue identity and legal resource states.
- Allocation owner, aliases, last consumers and the rule for safe reuse.
- Publication after producer writes/copy completion; consumer acquisition before reads; release after all consumers finish.
- Persistent weights and KV assignments, with position/mask inputs and update semantics.
- Source hashes, compiler/runtime versions and required capabilities.

Fence dependencies are nominated from the graph's cut edges and joins. Values published together on an ordered queue can share one completion signal; a fence per tensor is unnecessary. Separate independent queues require enough signals to cover their true dependencies. Reuse must not race a later invocation. The builder computes lifetimes and required storage from the specified invocation concurrency; do not introduce a ring or multiple versions without evidence they are required.

A portable contract describes semantics. A compiled plan binds that contract to verified physical devices/resources. Changing topology or shape beyond its bounds requires rebuilding the plan.

## Deliverable 4: prove the ORT/DML resource seam
Before promising a zero-host-wait executor, prove the installed ORT/DML interface can bind producer outputs and consumer inputs to owned D3D12 resources and queues. One DML EP/session is attached to one D3D12 device; cross-device edges are our contract.

Test one real nominated boundary using device tensors/I/O binding and custom-device/session APIs where available. Verify that session execution does not secretly force a host completion/readback at this boundary. GPU tracing and native API observations are acceptance evidence; an async method name is not evidence.

Distinguish two transport modes:
1. Consumer reads the shared host aperture directly, if its DirectML tensor/resource binding and hardware permit it.
2. GPU copy publishes through a shared cross-adapter resource/aperture and acquires into consumer-local storage.

Mode 2 is an explicit measured alternative, not a claim of direct aperture consumption. Neither mode permits managed arrays, CPU memcpy or CPU waits as the multi-device bridge. Do not assume a generic host pointer is a valid D3D12 tensor or that every cross-adapter heap is directly bindable by DML.

If ORT's public seam enforces incompatible host waits, record the exact call and trace. Then assess lowering the already nominated regions to DirectML compiled graphs plus a small D3D12 command recorder. Reuse ONNX/ORT transformation recipes and supported operators; do not silently commit to implementing all Gemma operators.

## Deliverable 5: static command replay
A minimal executor loads the compiled plan, creates persistent resources/sessions once, records legal command sequences and submits them. Queue-side Signal/Wait and barriers enforce every cut and join. The CPU may enqueue future work asynchronously; it may not wait, poll or drain queues to bridge devices.

There is no dynamic ready queue, placement policy or host dependency solver. The only runtime selection is a precompiled invocation/shape contract. Final user-visible result delivery is distinct from inter-device ordering and is instrumented separately.

Cancellation, errors and resource teardown belong outside the successful hot path. They cannot justify a blocking fallback during a boundary transfer.

## Module shape
Proposed separate projects:
- `DirectAI.Graph.Contracts`: immutable contracts and serialization; no ORT dependency.
- `DirectAI.Graph.Analysis`: ONNX dependency/liveness/seam analysis.
- `DirectAI.Graph.Profiling`: opt-in diagnostic execution and measurements.
- `DirectAI.Graph.Build`: partition emission and plan compilation.
- `DirectAI.Graph.D3D12`: owned resources/queues, fence and replay implementation.
- `DirectAI.Graph.Plugin`: discovery/invocation adapter.

Only dependency direction needed for those functions is allowed. Core and HTTP/pipe/CLI assemblies do not gain graph internals. Embed plugin manifests and declare external/native/model dependencies. Supply PS1 commands for inspect, nominate, build, validate and measure. PowerShell can construct the runtime directly without the host executable.

## Acceptance gates and order
1. Whole-graph analysis identifies all live values across a nominated real Gemma seam; reconstructed graph validates.
2. Two partitions on one device match unsplit logits/KV with documented dtype-aware tolerances, fixed inputs and multi-token checks. Compare prefill and decode separately.
3. Two V340 dies produce matching outputs while traces show no application CPU wait/poll/readback at their boundary. Include adversarial consumer-before-producer submission and safe resource reuse tests.
4. Same weights/shapes on one versus two dies: warm latency distribution, per-region GPU time, payload bytes, bridge latency, submission counts, peak VRAM and output error.
5. Four identical dies: capacity plus latency with resident KV; only then the P2000 heterogeneous compatibility test.
6. Publish paired JSON receipts and a reproducible command. Preserve unsuccessful results; no “unlock” declaration from a microbenchmark.

Stop at a failed gate and report the missing API or correctness property. First implementation scope is analysis and seam nomination, not a new serving framework or LLM backend.

## Existing work to reuse
- [DirectML CompileGraph](https://learn.microsoft.com/en-us/windows/win32/api/directml/nf-directml-idmldevice1-compilegraph): compile connected operator regions within a device.
- [ORT DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) and [device tensors](https://onnxruntime.ai/docs/performance/device-tensor.html): verify exact installed API support before implementation.
- [Rammer](https://www.usenix.org/system/files/osdi20-ma.pdf) / [NNFusion](https://github.com/microsoft/nnfusion): precedent for moving execution decisions to compilation.
- [Alpa](https://github.com/alpa-projects/alpa): study offline candidate profiling and partition construction; do not transplant its distributed runtime.
- [SliceGraph](https://arxiv.org/abs/2311.03703): inference graph partitioning with compute/communication costs.

These are architectural references, not evidence that their CUDA/distributed implementations already support our Windows DML path.
