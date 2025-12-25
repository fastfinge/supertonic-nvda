import os
import sys
import threading
import queue
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
		
		# Initialize settings with defaults
		self._voice = self._tts.voice_style_names[0] if self._tts.voice_style_names else "M1"
		# Default speed is 1.05. 
		# NVDA rate 27 maps to approx 1.05 with our mapping: 0.7 + (27/100)*1.3 = 1.051
		self._rate = 27 
		self._quality = 5
		
		self._job_queue = queue.Queue()
		self._generation = 0
		self._generation_lock = threading.Lock()
		self._stop_event = threading.Event()
		self._worker_thread = threading.Thread(target=self._worker, daemon=True)
		self._worker_thread.start()

	def _worker(self):
		while not self._stop_event.is_set():
			try:
				job = self._job_queue.get(timeout=0.5)
			except queue.Empty:
				continue
			
			generation, text, index_map, voice_name, rate, quality = job
			
			with self._generation_lock:
				if generation != self._generation:
					self._job_queue.task_done()
					continue
			
			try:
				self._process_job(generation, text, index_map, voice_name, rate, quality)
			except Exception:
				log.error("Error in Supertonic worker", exc_info=True)
				synthDoneSpeaking.notify()
			finally:
				self._job_queue.task_done()

	def _process_job(self, job_generation, text, index_map, voice_name, rate, quality):
		try:
			# Sanitize text and remap indices
			processor = self._tts.model.text_processor
			new_text = ""
			remap = [0] * (len(text) + 1)
			
			for i, char in enumerate(text):
				remap[i] = len(new_text)
				is_valid, _ = processor.validate_text(char)
				if is_valid:
					new_text += char
				# If invalid, it is skipped, and remap[i] remains pointing to current len(new_text)
				# which is the position effectively "after" the previous valid char.
			
			remap[len(text)] = len(new_text)
			
			# Rebuild index map with new offsets
			new_index_map = []
			for offset, idx in index_map:
				# Clamp offset to bounds just in case
				if offset > len(text):
					offset = len(text)
				new_offset = remap[offset]
				new_index_map.append((new_offset, idx))
			
			text = new_text
			index_map = new_index_map

			voice_style = self._tts.get_voice_style(voice_name)
			speed = 0.7 + (rate / 100.0) * (2.0 - 0.7)
			
			# Check cancellation before synthesis
			with self._generation_lock:
				if self._generation != job_generation:
					return

			wav, _, dur_lists = self._tts.synthesize(
				text,
				voice_style=voice_style,
				speed=speed,
				total_steps=quality,
				max_chunk_length=300,
				silence_duration=0.1,
				return_alignment=True
			)
			
			all_durations = np.concatenate(dur_lists)
			bytes_per_sec = self._tts.sample_rate * 2
			
			audio_data = np.clip(wav.squeeze() * 32767, -32768, 32767).astype(np.int16).tobytes()
			
			# Check cancellation before feeding
			with self._generation_lock:
				if self._generation != job_generation:
					return
			
			# Calculate byte offsets and group indices
			cum_durations = np.cumsum(all_durations)
			indices_by_offset = {}
			audio_len = len(audio_data)
			
			for char_offset, index in index_map:
				if char_offset >= len(cum_durations):
					# Map to end of audio
					target_time = cum_durations[-1] if len(cum_durations) > 0 else 0.0
				elif char_offset == 0:
					target_time = 0.0
				else:
					target_time = cum_durations[char_offset - 1]
				
				target_byte = int(target_time * bytes_per_sec)
				target_byte = target_byte - (target_byte % 2)
				
				# Clamp to audio length to avoid feeding empty chunks past end
				if target_byte > audio_len:
					target_byte = audio_len
				
				if target_byte not in indices_by_offset:
					indices_by_offset[target_byte] = []
				indices_by_offset[target_byte].append(index)
			
			sorted_offsets = sorted(indices_by_offset.keys())
			last_fed_byte = 0
			
			for offset in sorted_offsets:
				# Check cancellation
				with self._generation_lock:
					if self._generation != job_generation:
						return
				
				indices = indices_by_offset[offset]
				
				if offset == 0:
					# Fire immediately
					for idx in indices:
						synthIndexReached.notify(index=idx)
					continue
				
				# Feed audio up to this offset
				chunk_len = offset - last_fed_byte
				if chunk_len > 0:
					chunk = audio_data[last_fed_byte:offset]
					
					# Define callback
					def on_done(idxs=indices):
						for i in idxs:
							synthIndexReached.notify(index=i)
					
					self._player.feed(chunk, onDone=on_done)
					last_fed_byte = offset
				else:
					for idx in indices:
						synthIndexReached.notify(index=idx)
			
			# Feed remaining audio
			if last_fed_byte < len(audio_data):
				chunk = audio_data[last_fed_byte:]
				self._player.feed(chunk)
			
			# Wait for playback to finish
			self._player.idle()
			synthDoneSpeaking.notify()

		except Exception as e:
			raise e

	def speak(self, speechSequence):
		text = ""
		index_map = [] 

		for item in speechSequence:
			if isinstance(item, str):
				text += item
			elif isinstance(item, IndexCommand):
				index_map.append((len(text), item.index))

		if not text.strip():
			synthDoneSpeaking.notify()
			return

		with self._generation_lock:
			generation = self._generation

		# Push to queue
		self._job_queue.put((generation, text, index_map, self._voice, self._rate, self._quality))

	def cancel(self):
		with self._generation_lock:
			self._generation += 1
		
		# Clear queue
		while not self._job_queue.empty():
			try:
				self._job_queue.get_nowait()
				self._job_queue.task_done()
			except queue.Empty:
				break
		
		self._player.stop()

	def pause(self, switch):
		self._player.pause(switch)

	def terminate(self):
		self._stop_event.set()
		self._player.stop()
		self._player.close()
		self._worker_thread.join()

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
