---
license: openrail
language:
- en
pipeline_tag: text-to-speech
---

# Supertonic — Lightning Fast, On-Device TTS

**Supertonic** is a lightning-fast, on-device text-to-speech system designed for **extreme performance** with minimal computational overhead. Powered by ONNX Runtime, it runs entirely on your device—no cloud, no API calls, no privacy concerns.

> 🎧 **Try it now**: Experience Supertonic in your browser with our [**Interactive Demo**](https://huggingface.co/spaces/Supertone/supertonic#interactive-demo), or [**Hugging Face app**](https://huggingface.co/spaces/akhaliq/supertonic) or get started with pre-trained models from [**Hugging Face Hub**](https://huggingface.co/Supertone/supertonic)

> 🛠 **GitHub Repository**  
> To use Supertonic most easily, visit the official GitHub repository:  
> https://github.com/supertone-inc/supertonic  
> You’ll find multi-language example codes.

### Table of Contents

- [Why Supertonic?](#why-supertonic)
- [Language Support](#language-support)
- [Getting Started](#getting-started)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)

## Why Supertonic?

- **⚡ Blazingly Fast**: Generates speech up to **167× faster than real-time** on consumer hardware (M4 Pro)—unmatched by any other TTS system
- **🪶 Ultra Lightweight**: Only **66M parameters**, optimized for efficient on-device performance with minimal footprint
- **📱 On-Device Capable**: **Complete privacy** and **zero latency**—all processing happens locally on your device
- **🎨 Natural Text Handling**: Seamlessly processes numbers, dates, currency, abbreviations, and complex expressions without pre-processing
- **⚙️ Highly Configurable**: Adjust inference steps, batch processing, and other parameters to match your specific needs
- **🧩 Flexible Deployment**: Deploy seamlessly across servers, browsers, and edge devices with multiple runtime backends.


## Language Support

We provide ready-to-use TTS inference examples across multiple ecosystems:

| Language/Platform | Path | Description |
|-------------------|------|-------------|
| [**Python**] | `py/` | ONNX Runtime inference |
| [**Node.js**] | `nodejs/` | Server-side JavaScript |
| [**Browser**] | `web/` | WebGPU/WASM inference |
| [**Java**] | `java/` | Cross-platform JVM |
| [**C++**] | `cpp/` | High-performance C++ |
| [**C#**] | `csharp/` | .NET ecosystem |
| [**Go**] | `go/` | Go implementation |
| [**Swift**] | `swift/` | macOS applications |
| [**iOS**] | `ios/` | Native iOS apps |
| [**Rust**] | `rust/` | Memory-safe systems |
| [**Flutter**] | `flutter/` | Cross-platform apps |

> For detailed usage instructions, please refer to the README.md in each language directory.

## Getting Started

First, clone the repository:

```bash
git clone https://github.com/supertone-inc/supertonic.git
cd supertonic
```

### Prerequisites

Before running the examples, download the ONNX models and preset voices, and place them in the `assets` directory:

```bash
git clone https://huggingface.co/Supertone/supertonic assets
```

> **Note:** The Hugging Face repository uses Git LFS. Please ensure Git LFS is installed and initialized before cloning or pulling large model files.
> - macOS: `brew install git-lfs && git lfs install`
> - Generic: see `https://git-lfs.com` for installers


### Technical Details

- **Runtime**: ONNX Runtime for cross-platform inference (CPU-optimized; GPU mode is not tested)
- **Browser Support**: onnxruntime-web for client-side inference
- **Batch Processing**: Supports batch inference for improved throughput
- **Audio Output**: Outputs 16-bit WAV files

## Performance

We evaluated Supertonic's performance (with 2 inference steps) using two key metrics across input texts of varying lengths: Short (59 chars), Mid (152 chars), and Long (266 chars).

**Metrics:**
- **Characters per Second**: Measures throughput by dividing the number of input characters by the time required to generate audio. Higher is better.
- **Real-time Factor (RTF)**: Measures the time taken to synthesize audio relative to its duration. Lower is better (e.g., RTF of 0.1 means it takes 0.1 seconds to generate one second of audio).

### Characters per Second
| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 912 | 1048 | 1263 |
| **Supertonic** (M4 pro - WebGPU) | 996 | 1801 | 2509 |
| **Supertonic** (RTX4090) | 2615 | 6548 | 12164 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 144 | 209 | 287 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 37 | 55 | 82 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 12 | 18 | 24 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 38 | 64 | 92 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 104 | 107 | 117 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 37 | 42 | 47 |

> **Notes:**  
> `API` = Cloud-based API services (measured from Seoul)  
> `Open` = Open-source models  
> Supertonic (M4 pro - CPU) and (M4 pro - WebGPU): Tested with ONNX  
> Supertonic (RTX4090): Tested with PyTorch model  
> Kokoro: Tested on M4 Pro CPU with ONNX  
> NeuTTS Air: Tested on M4 Pro CPU with Q8-GGUF

### Real-time Factor

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.015 | 0.013 | 0.012 |
| **Supertonic** (M4 pro - WebGPU) | 0.014 | 0.007 | 0.006 |
| **Supertonic** (RTX4090) | 0.005 | 0.002 | 0.001 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 0.133 | 0.077 | 0.057 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 0.471 | 0.302 | 0.201 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 1.060 | 0.673 | 0.541 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 0.372 | 0.206 | 0.163 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 0.144 | 0.124 | 0.126 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 0.390 | 0.338 | 0.343 |

<details>
<summary><b>Additional Performance Data (5-step inference)</b></summary>

<br>

**Characters per Second (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 596 | 691 | 850 |
| **Supertonic** (M4 pro - WebGPU) | 570 | 1118 | 1546 |
| **Supertonic** (RTX4090) | 1286 | 3757 | 6242 |

**Real-time Factor (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.023 | 0.019 | 0.018 |
| **Supertonic** (M4 pro - WebGPU) | 0.024 | 0.012 | 0.010 |
| **Supertonic** (RTX4090) | 0.011 | 0.004 | 0.002 |

</details>

## License

This project’s sample code is released under the MIT License. - see the [LICENSE](https://github.com/supertone-inc/supertonic?tab=MIT-1-ov-file) for details.

The accompanying model is released under the OpenRAIL-M License. - see the [LICENSE](https://huggingface.co/Supertone/supertonic/blob/main/LICENSE) file for details.

This model was trained using PyTorch, which is licensed under the BSD 3-Clause License but is not redistributed with this project. - see the [LICENSE](https://docs.pytorch.org/FBGEMM/general/License.html) for details.

Copyright (c) 2025 Supertone Inc.