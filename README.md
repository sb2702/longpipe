# Longpipe
Hardware accelerated real time media processing in the browser

### Problem
If you want to real real-time AI media processing in the browser (or things like Virtual Backgrouns), one of the main open source options is [Mediapipe](https://github.com/google-ai-edge/mediapipe). Mediapipe has some great open source models, but they are all deployed in WebAssembly, and using them effectively usually requires hardware accelerated pre-processing and post-processing

![1_kOOB7swH0hEf0BQaH4p6TQ (1)](https://github.com/user-attachments/assets/62932072-c5b2-445e-b5d8-a2bd5bb72920)


In 2021, I built an SDK to implement [hardware acceleraed networks directly in WebGL](https://medium.com/vectorly/building-a-more-efficient-background-segmentation-model-than-google-74ecd17392d5), which proved much more efficient than the MediaPipe implementation. This proved very popular but it was a commercial SDK and the company was acquired and the technology was never exposed.

In 2022, Google Meet adopted a similar approach by writing their own [hardware accelerated networks](https://research.google/blog/high-definition-segmentation-in-google-meet/) but this was never open sourced. 


As of late 2025, Mediapipe is still the "state" of the art, and nothing better has come along.

### Solution

With the release of WebGPU, and the coming of WebNN, it is entirely possible to build efficient implementations of popular real-time media-processing features like Background Segmentation (Virtual Backgrounds) and Audio Filtering (Background Noise removal), combining hardware accelerated neural networks written in WebGPU and WebNN, along with efficient pre-/post-processing so that developers can just implement state of the art features like Virtual Backgrounds and noise removal without worrying about the details
