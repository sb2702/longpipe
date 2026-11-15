# MediaPipe Selfie Segmenter weights: NOTICE

The files `mediapipe_selfie_segmenter.f16.bin` and
`mediapipe_selfie_segmenter_landscape.f16.bin` (served from
`https://cdn.longpipe.dev/models/mediapipe/selfie/v1/`) contain the trained weights of
Google's **MediaPipe Selfie Segmenter**, square (256×256) and landscape (144×256)
variants.

- Copyright Google LLC. Authors per the model card: Tingbo Hou, Siargey Pisarchyk,
  Karthik Raveendran (Google).
- Licensed under the **Apache License, Version 2.0**: see `LICENSE` in this directory,
  or https://www.apache.org/licenses/LICENSE-2.0.
- License source: the official model card,
  https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf
  ("LICENSED UNDER: Apache License, Version 2.0"; it covers both the general and the
  landscape model), linked from
  https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#selfie-model

## Original files

| Variant   | Source                                                                                                                      | sha256                                                             |
|-----------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| square    | https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite                     | `191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b` |
| landscape | https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite | `490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e` |

## Modifications

Format conversion only, by `scripts/convert_mediapipe_selfie.py` in
https://github.com/sb2702/longpipe: the convolution, depthwise and transposed-convolution
weights and biases were extracted from the `.tflite` flatbuffers, repacked into
Longpipe's GPU shader layout (4×4 channel blocks, zero-padded to multiples of 4
channels) and written in Longpipe's binary weight-pack format. The numeric values are
the original float16 values, unchanged. The final layer of the landscape model
(`conv2d_transpose`) is stored under the square model's name (`segment`). No
retraining, fine-tuning or quantization.

The network graph that runs these weights (`src/model/networks/mediapipe_selfie.ts`)
is an independent re-implementation, part of Longpipe (MIT). These weight files are
NOT covered by Longpipe's MIT `WEIGHTS_LICENSE`; they remain under Apache-2.0.
