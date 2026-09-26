# Deferred work

Updated 2026-09-19 under D027. ROADMAP.md is the active scope. Preserve existing scaffolds/configs;
these items are not automatic work orders. Reopen only after the stated prerequisite and a bounded
hypothesis. Existing run/publication approval requirements remain.

| Work | Reopen when | Rationale for deferral |
| --- | --- | --- |
| 100M training/export | Core tiny path and controls are measured | Adds cost before the primary deployment question is answered |
| Full MLX port and runtime sweep | Core ML frontier exists and a second runtime resolves a decision | Maintaining another encoder/decoder implementation is substantial work |
| Distillation | Direct tiny and practical controls establish a gap worth addressing | FloodDistill already overlaps; define the distinct hypothesis first |
| Custom reconstruction PTQ or QAT | Vendor baselines expose a specific failure | Existing algorithms are not novelty merely because reimplemented |
| Compact decoder ablation | Component profile motivates it | Parameter overhead is recorded; latency dominance is not yet measured |
| Wildfire / landslides | Flood comparison is complete | A second task cannot rescue missing evidence on the first |
| Another compact pretrained family | Core comparisons complete | Useful broader context, not a reason to delay a complete first result |
| Physical phone/tablet study | Full segmentation artifact is ready and device available | Required for mobile claims, not required for an honest Mac-only study |
| External flood dataset | Split/label/preprocessing mapping is audited | Useful generalization evidence; incompatible scores must stay separate |

Outside the core study: full iOS application, satellite deployment, live imagery delivery,
Sentinel-1/SAR, temporal inference, 600M models, extra export stacks, pretraining-level distillation,
weakly labeled pool expansion and PyPI packaging. Energy measurement and Xcode repair retain their
existing permission requirements. Upstreaming the converter fix could help others but is not on
the immediate path. Historical resize/tiling comparisons are distinct from the approved native
protocol; no silent per-runtime geometry changes.
