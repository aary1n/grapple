## Grapple Enterprise Roadmap

### Goals
- Build an enterprise-ready, secure, low-latency OS-level gesture-to-CAD control layer.
- Provide first-class integrations for SolidWorks (first), then AutoCAD/Inventor/NX.
- Meet enterprise requirements for deployment, observability, compliance, and manageability.

### Milestones
- 0.9 Prototype Hardening (stability, basic observability, input robustness)
- 1.0 Enterprise MVP (Windows Service, gRPC over Named Pipes + shared memory, SolidWorks add-in, MSIX installer)
- 1.1 Integrations Pack (AutoCAD plugin, mapping engine policies, update channel)
- 1.2 Platform Hardening (security, performance to <50 ms p99, governance/licensing)

---

## 0.9 Prototype Hardening

### Architecture and IPC (P0)
- [x] Replace `mouse_event` with `SendInput` for robust input injection (DPI-aware).
- [x] Add DPI scaling and multi-monitor normalization for cursor motion (Phase 4: `SendInput` + `MOUSEEVENTF_VIRTUALDESK`, `PerMonitorV2` manifests, `DisplayInfo` abstraction).
- [ ] Implement absolute positioning path and client-area normalization for active CAD window.

### Performance and Latency (P0)
- [x] Process latest frame only (drop backlog); introduce backpressure-aware queue (Phase 1: `AtomicMailbox` LIFO + backpressure detection in `WebcamCaptureNode`).
- [x] Pre-allocate buffers; warm up detection on startup (Phase 2: FlatBuffer pre-allocated read buffers, `stackalloc` in telemetry flush).
- [x] Add exponential smoothing/Kalman filter for motion stability (configurable).

### Resilience and Health (P0)
- [ ] Formalize health model: cameraOk, handOk, detectorOk with thresholds.
- [ ] Add circuit breaker for detector failures and exponential backoff on restarts.
- [ ] UI health banner and basic diagnostics view.

### Observability (P0)
- [x] Structured logging (Phase 4: `GrappleLogger` — JSON-lines, per-category throttle, configurable `MinLevel`). Note: custom lightweight logger, not Serilog — zero external dependencies in `Grapple.Core`.
- [x] Minimal metrics: FPS, E2E latency (P50/P95/P99), dropped frames, GC counts (Phase 4: `TelemetryCollector` with lock-free counters, 10Hz FlatBuffer flush to `TelemetryArena`).
- [ ] Crash handling and minidump integration (WER or Sentry/Raygun – choose one).

### Security and Compliance (P1)
- [ ] Document data flow: frames stay on-device; no cloud dependency.
- [ ] Pin Python and MediaPipe versions; create dependency lock manifest.

### Configuration (P0) — Added Phase 3
- [x] Externalize all hardcoded constants into shared `grapple_config.json` (56+ constants).
- [x] Config-driven C# pipeline (`GrappleConfig` + `GrappleConfigLoader`).
- [x] Config-driven Python detector (loads from same JSON).
- [x] `TelemetryCollectionConfig` and `LoggingConfig` sections.

---

## 1.0 Enterprise MVP

### Architecture and IPC (P0)
- [x] Replace stdio JSON with gRPC over Named Pipes (Windows).
- [x] Define `.proto` (versioned): DetectRequest/Response, Health, Version, Capabilities.
- [ ] Remove base64 transfer; adopt zero-copy shared memory for frames (CreateFileMapping/MapViewOfFile).
- [ ] Send only shared memory handle + metadata (width, height, stride, pixel format, frameId).
- [ ] Make detector a standalone Windows Service with watchdog and resource limits.
- [ ] Expose health and version endpoints from service.

### Detector Runtime Strategy (P0)
- [ ] Short-term: package Python MediaPipe; embed Python (Windows embeddable) or freeze with PyInstaller/pyoxidizer.
- [ ] Pin dependencies and hashes; sign the detector binary.
- [ ] Long-term Options (evaluate + decision record):
  - [ ] A: MediaPipe C++ tasks via native DLL + P/Invoke/C++/CLI.
  - [ ] B: Convert to ONNX and run with ONNX Runtime (CPU/CUDA/DirectML) in .NET.

### Input Injection Reliability (P0)
- [ ] Switch all pointer events to `SendInput` (relative + absolute paths).
- [ ] Add support for elevated/system apps (pointer injection API) – spike and feasibility.
- [ ] Research/plan a Virtual HID Miniport driver for true device-level events (design doc).

### CAD-Native Integrations (P0)
- [ ] SolidWorks .NET add-in: connect to local Named Pipe; map gestures → pan/orbit/zoom/select.
- [ ] Establish gesture-to-command schema (versioned) and default SolidWorks profile.
- [ ] Add focus/activation routing so commands go to the active SolidWorks window.

### Installer, Packaging, Updates (P0)
- [ ] MSIX (preferred) or MSI (WiX) per-machine installer.
- [ ] Bundle: self-contained .NET app, detector runtime, models, SolidWorks add-in.
- [ ] Register/start Windows Service; configure recovery; create Event Log source.
- [ ] Silent install switches; rollback on failure.
- [ ] Code signing with EV certificate for installers and executables.

### Configuration and Policy (P0)
- [ ] Central config at `%ProgramData%\\Grapple\\config.json`.
- [ ] Policy overrides via HKLM (ADMX template) for camera, telemetry, updates, per-app enablement, gesture sets.
- [ ] Profile precedence: machine → user → app.

### Observability (P1)
- [ ] ETW provider for high-volume tracing (optional build flag).
- [ ] OpenTelemetry metrics exporter (local only by default; allow OTLP endpoint if permitted).
- [ ] “Collect diagnostics” zip from UI: logs, config, versions, health snapshot.

---

## 1.1 Integrations Pack

### CAD-Native Integrations (P0)
- [ ] AutoCAD plugin (.NET) with gesture mapping for pan/orbit/zoom/select/draw tools.
- [ ] Inventor/NX: feasibility assessment and MVP mapping.
- [ ] Mapping engine: profiles per app with policy-controlled overrides.

### Updates and Channels (P0)
- [ ] Auto-update service with staged rings (Dev/Canary/Stable).
- [ ] Delta updates and offline bundles for air-gapped sites.

### Compatibility and UX (P1)
- [ ] DPI scaling and multi-monitor transforms validated across setups.
- [ ] HDR/advanced display modes tested.
- [ ] Calibration wizard; left/right handedness; visual feedback toggles.

### Security and Compliance (P1)
- [ ] SBOM generation and SCA scanning for .NET + Python artifacts.
- [ ] Reproducible build steps and artifact verification.

---

## 1.2 Platform Hardening

### Performance and Latency (P0)
- [ ] Camera capture via Media Foundation with GPU texture path (avoid extra BGR↔RGB copies).
- [ ] GPU inference (CUDA/DirectML) with pinned models; measure p50/p95/p99 latency.
- [ ] Thread affinity and QoS tuning; isolate UI from detector workload.

### Resilience and Health (P0)
- [ ] Soak tests (24–72h) for leaks and restart resilience.
- [ ] Watchdog elevates incidents to Windows Event Log with actionable codes.

### Security, Privacy, Compliance (P0)
- [ ] On-disk encryption for cached models and secure temp.
- [ ] Least-privilege service account with no network access; ACLs on pipes/shmem.
- [ ] Policy pack (ADMX) finalized and documented.

### Governance and Licensing (P1)
- [ ] License model (per-seat, offline activation, or SSO entitlements) – implement chosen model.
- [ ] Admin CLI/dashboard for activation, assigning profiles, and locking settings.
- [ ] Legal review of model redistribution licenses (MediaPipe/ONNX).

---

## QA and Certification

### Automated Testing (P0)
- [x] Unit tests for gesture state machine and filters (55 tests: protocol compat, config, display/coordinate mapping, mailbox, telemetry, integration).
- [ ] Integration tests: synthetic frames → landmarks → command outputs.
- [ ] Hardware-in-the-loop: virtual camera feed + verify `SendInput` outcomes.

### Test Matrix (P0)
- [ ] Webcams (top 10 models), CPU-only, NVIDIA/AMD/iGPU.
- [ ] RDP/VDI scenarios and elevated app interactions.

### Release Criteria (P0)
- [ ] p99 E2E latency < 50 ms on reference hardware.
- [ ] No data leaves device; verified by inspection and telemetry config.
- [ ] Installer rollback-safe; service recovers from crashes.

---

## Documentation and Support

### Enterprise Docs (P0)
- [ ] Deployment guide (GPO, SCCM/Intune), silent install, updates.
- [ ] App integration guides: SolidWorks first; then AutoCAD/Inventor/NX.
- [ ] Gesture mapping templates and best practices.

### IT Runbook (P0)
- [ ] Health checks, log locations, Event Log sources, common remediation steps.
- [ ] Diagnostics bundle workflow and privacy posture.

---

## Decision Records (keep current)
- [ ] Choice of IPC transport (Named Pipes + gRPC) – rationale, alternatives.
- [ ] Detector runtime (Python vs native/ONNX) – migration plan.
- [ ] Input path (SendInput vs pointer injection vs HID driver) – roadmap.

---

## Tracking
- Status keys: P0 = must-have for milestone, P1 = nice-to-have, P2 = stretch.
- Progress is tracked via checkboxes; link issues/PRs to items as created.


