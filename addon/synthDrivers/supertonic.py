import os
import sys
from pathlib import Path

# Add the libs directory to sys.path so we can import supertonic and its dependencies
libs_path = os.path.join(os.path.dirname(__file__), "libs")
if libs_path not in sys.path:
	sys.path.insert(0, libs_path)

import numpy as np
import synthDriverHandler
from logHandler import log
import nvwave
import supertonic
from synthDriverHandler import synthIndexReached, synthDoneSpeaking
from autoSettingsUtils.driverSetting import NumericDriverSetting
from speech.commands import IndexCommand

class SynthDriver(synthDriverHandler.SynthDriver):
	"""
	NVDA Synth Driver for Supertonic TTS.
	"""
	name = "supertonic"
	description = _("Supertonic")

	@classmethod
	def check(cls):
		return True

	def __init__(self):
		super().__init__()
		model_dir = Path(__file__).parent / "models"
		try:
			self._tts = supertonic.TTS(model_dir=model_dir, auto_download=False)
		except Exception:
			log.error("Failed to initialize Supertonic TTS", exc_info=True)
			raise RuntimeError("Supertonic initialization failed")

		self._player = nvwave.WavePlayer(
			channels=1,
			samplesPerSec=self._tts.sample_rate,
			bitsPerSample=16
		)
		self._player.syncFunc = self._onWavePlayerSync
		self._index_queue = []

		# Initialize settings with defaults
		self._voice = self._tts.voice_style_names[0] if self._tts.voice_style_names else "M1"
		# Default speed is 1.05. 
		# NVDA rate 27 maps to approx 1.05 with our mapping: 0.7 + (27/100)*1.3 = 1.051
		self._rate = 27 
		self._quality = 5

	def _onWavePlayerSync(self, *args):
		if self._index_queue:
			index = self._index_queue.pop(0)
			synthIndexReached.notify(index=index)

	def speak(self, speechSequence):
		# Reconstruct text and track index positions
		text = ""
		index_map = [] # List of (char_offset, index)

		for item in speechSequence:
			if isinstance(item, str):
				text += item
			elif isinstance(item, IndexCommand):
				index_map.append((len(text), item.index))

		if not text.strip():
			synthDoneSpeaking.notify()
			self._player.idle()
			return

		try:
			voice_style = self._tts.get_voice_style(self._voice)
			# Convert NVDA rate (0-100) to Supertonic speed (0.7-2.0)
			speed = 0.7 + (self._rate / 100.0) * (2.0 - 0.7)

			# Supertonic synthesize returns (waveform, duration)
			# We request alignment data
			wav, _, dur_lists = self._tts.synthesize(
				text,
				voice_style=voice_style,
				speed=speed,
				total_steps=self._quality,
				max_chunk_length=100, # Try to keep it in one chunk
				silence_duration=0.1,
				return_alignment=True
			)

			# Concatenate all durations
			all_durations = np.concatenate(dur_lists)
			bytes_per_sec = self._tts.sample_rate * 2

			# Prepare audio data
			audio_data = np.clip(wav.squeeze() * 32767, -32768, 32767).astype(np.int16).tobytes()

			# Feed audio and syncs
			last_fed_byte = 0
			index_map.sort(key=lambda x: x[0])
			cum_durations = np.cumsum(all_durations)

			for char_offset, index in index_map:
				if char_offset >= len(cum_durations):
					continue

				if char_offset == 0:
					target_time = 0.0
				else:
					target_time = cum_durations[char_offset - 1]

				target_byte = int(target_time * bytes_per_sec)
				# Align to block align (2 bytes)
				target_byte = target_byte - (target_byte % 2)

				if target_byte > last_fed_byte:
					chunk = audio_data[last_fed_byte:target_byte]
					self._player.feed(chunk)
					last_fed_byte = target_byte

				self._index_queue.append(index)
				self._player.sync()

			# Feed remaining audio
			if last_fed_byte < len(audio_data):
				self._player.feed(audio_data[last_fed_byte:])

		except Exception:
			log.error("Supertonic synthesis failed", exc_info=True)
			synthDoneSpeaking.notify()

	def cancel(self):
		self._index_queue.clear()
		self._player.stop()
		self._player.idle()


	def pause(self, switch):
		self._player.pause(switch)

	def terminate(self):
		self._player.stop()
		self._player.close()

	def _get_availableVoices(self):
		voices = {}
		for name in self._tts.voice_style_names:
			voices[name] = synthDriverHandler.VoiceInfo(name, name)
		return voices

	def _get_voice(self):
		return self._voice

	def _set_voice(self, value):
		self._voice = value

	def _get_rate(self):
		return self._rate

	def _set_rate(self, value):
		self._rate = value

	def _get_quality(self):
		return self._quality

	def _set_quality(self, value):
		self._quality = value

	supportedSettings = (
		NumericDriverSetting("quality", _("Speech Quality Control"), 1, 100, 5),
		synthDriverHandler.SynthDriver.VoiceSetting(),
		synthDriverHandler.SynthDriver.RateSetting(),
	)

	supportedNotifications = {synthIndexReached, synthDoneSpeaking}