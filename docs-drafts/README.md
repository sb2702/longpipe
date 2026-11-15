# Website doc drafts for `model: 'mediapipe-selfie'`

Not applied to `longpipe-website` (per instructions). To apply, from the website repo root:

```sh
git apply ../longpipe/docs-drafts/website-docs.patch        # api, self-hosting, model-layer, presets
cp ../longpipe/docs-drafts/models.mdx src/pages/docs/models.mdx   # new page
```

Also: add `models` to the docs nav, and regenerate `llms.txt` / `llms-full.txt` if they're built from the pages.

Check before publishing: the speed sentence in `models.mdx` ("`xs` ~1.7 ms vs ~2.6–3.1 ms") comes from one quick
3-run session on the M4 (model-only mode, WebGPU, `benchmarks` harness). Replace it with the benchmarks branch's
proper rows once they include the MediaPipe-in-Longpipe pipelines.

## `bench/`

- `2026-11-15-apple-m4-mediapipe-in-longpipe-quick.json`: unedited `npm run bench -- --headless --samples 360p,720p
  --modes transcode,model --runs 3` output from the `benchmarks` harness on the M4 (machine was also an idle
  upscaler worker; load avg ~4), with the MediaPipe-in-Longpipe pipelines added.
- `benchmarks-pipelines.patch`: those pipelines (`lp-mediapipe-{webgpu,webgl}-{square,landscape}-{f32,f16}`), as a
  patch to `benchmarks/performance/web/src/pipelines.js` on the `benchmarks` branch. Needs this branch's `longpipe`
  build and the two `.f16.bin` files in `web/public/assets/models/`.
- `smoke.html` + `smoke.mjs`: the headless-Chrome smoke test (real `EffectsPipeline` with `model: 'mediapipe-selfie'`
  on WebGPU and, with `--nowebgpu`, WebGL2; plus alpha vs `@mediapipe/tasks-vision` on three frames). Drop into
  `benchmarks/performance/web/` and `tools/`.
