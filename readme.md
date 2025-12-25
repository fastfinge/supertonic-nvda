# Supertonic TTS for NVDA

Warning: this is alpha quality software. Expect issues.

This add-on provides a synthesizer driver for the Supertonic text-to-speech engine in NVDA.
Supertonic is a high-performance, on-device TTS powered by ONNX Runtime. However, while extremely fast for its sound quality, it's still not quite fast enough for realtime use. Consider using this for say all, and nothing else.

## Features

- High-quality, lightning-fast speech synthesis.
- On-device processing (no internet required).
- Multiple voice styles.
- Control over speech speed and quality.

## Issues

- Speed only pretends to work. Increasing speed makes it skip words instead of actually speaking faster.
- Pitch can't be changed
- I suspect we don't need to bundle quite as many Python packages as we do, but NVDA includes weird versions of things and I'm scared

## Settings

The following settings are available in NVDA's Speech settings dialog when Supertonic is selected as the synthesizer:

### Voice
Choose from several available voice styles (M1-M5, F1-F5).

### Rate (Speech Speed Control)
Adjust the speed of the speech.

### Speech Quality Control
Adjust the number of synthesis steps (1-100). Higher values result in better quality but slower synthesis. The default value of 5 provides an excellent balance between speed and quality.

## Requirements

- NVDA 2026.1 or later.
- Windows 10/11 (x64).