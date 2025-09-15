Save MP3 node for ComfyUI

<img width="331" height="222" alt="image" src="https://github.com/user-attachments/assets/f5c17ac3-b3dc-4721-a133-1eaedc15a85a" />

I felt the comfy-core node lacked some functionality so I made one with

- Audio input: accepts common formats used by audio nodes
- File path: choose output subfolder or absolute path on any drive
- Filename prefix: like the comfy-core node
- Bitrate mode: variable, constant, average (libmp3lame)
- Quality: low, medium, high (mapped per mode)
- Audio output for preview
  

Installation
1) Open CMD in your ComfyUI `custom_nodes/` directory.
2) Clone or copy
   ´´´git clone https://github.com/Dehypnotic/comfyui-save-mp3.git´´´
   
   
4) Backend options (no system install required):
   - Auto-download via imageio-ffmpeg: `pip install imageio-ffmpeg`. The first run downloads a static ffmpeg and caches it. No PATH changes needed.
   - Drop-in ffmpeg: place `ffmpeg.exe` (Windows) or `ffmpeg` (Linux/macOS) in `enhanced_save_audio_mp3/bin/`. The node will auto-detect it.
   - Or install `lameenc` via pip: `pip install lameenc`.
   - Or use a system-wide ffmpeg available on PATH (set `FFMPEG_PATH`/`FFMPEG_BINARY` to override).
5) Restart ComfyUI.

Node
- Class name: `SaveAudioMP3Enhanced`
- Category: `audio`

Inputs
- `audio` (AUDIO): waveform plus sample rate. Supported shapes:
  - Tuple `(np.ndarray, sample_rate)` or `(sample_rate, np.ndarray)`
  - Dict with keys: `samples`/`waveform`/`audio` and `sample_rate`/`sr`
  - Bare `np.ndarray` is not recommended (sample rate required). Arrays can be `(T,)`, `(T, C)` or `(C, T)`.
- `file_path` (STRING): subfolder under ComfyUI `output/` or an absolute folder path.
- `filename_prefix` (STRING): filename prefix, timestamp is appended.
- `bitrate_mode` (CHOICE): `variable`, `constant`, `average`.
- `quality` (CHOICE): `low`, `medium`, `high`.

Saving
- The node always saves to `file_path`.
- The UI player only appears if the file is saved under ComfyUI's `output/` directory.

Bitrate/quality mapping
- Variable (VBR): high→`-q:a 0`, medium→`-q:a 4`, low→`-q:a 7`
- Constant (CBR): high→`320k`, medium→`192k`, low→`128k`
- Average (ABR): high→`256k`, medium→`192k`, low→`160k` (uses `-abr 1`)

Notes
- Encoding uses lameenc (if installed) or ffmpeg. ffmpeg can be discovered via a bundled `bin/ffmpeg`, a system PATH, or auto-downloaded by `imageio-ffmpeg`.
- If neither lameenc nor any ffmpeg is found, the node raises a clear error with guidance.
- UI playback requires files to be under ComfyUI's output directory; when saving outside it, no inline player is produced.
- The node returns the audio for downstream processing and provides UI metadata so ComfyUI can render an audio player. If your ComfyUI build expects a different UI payload key for audio, tell me and I’ll tweak it.

FAQ / Customization
- Different filename pattern: adjust `_next_filename()` in `enhanced_save_audio_mp3/save_audio_mp3.py`.
- Default folders: change `_base_output_dir()` or the default `file_path` in `INPUT_TYPES`.
- If your audio arrays are outside [-1,1] or not float/int16, the node normalizes/clips and converts to 16‑bit PCM before encoding.

