## Save MP3 node for ComfyUI

Simple, flexible MP3 saver with bitrate options and handy path/filename templates.

Features
- Audio input: accepts common formats used by audio nodes
- File path: ComfyUI output subfolder or absolute path (any drive)
- Filename prefix
- Bitrate mode: variable, constant, average
- Quality: low, medium, high (mapped per mode)
- Outputs: `AUDIO` and `STRING` (bitrate info summary)
- Output node: can terminate a graph

Installation
1) Go to the `custom_nodes/` directory in ComfyUI.
2) Clone or copy the `comfyui-save-mp3` folder:
   ```bash
   git clone https://github.com/Dehypnotic/comfyui-save-mp3.git
   ```
3) Restart ComfyUI.

Optional backends (no system install required)
- Auto-download ffmpeg: `pip install imageio-ffmpeg` (first run caches a static ffmpeg)
- Drop-in ffmpeg: place `ffmpeg`/`ffmpeg.exe` in a `bin/` folder next to the node
- Or install `lameenc`: `pip install lameenc`

Backend preference: uses ffmpeg when available; otherwise falls back to `lameenc`.

Bitrate/quality mapping
- Variable (VBR): high → `-q:a 0` (~245 kbps), medium → `-q:a 4` (~165 kbps), low → `-q:a 7` (~100 kbps)
- Constant (CBR): high → `320k`, medium → `192k`, low → `128k`
- Average (ABR): high → `256k`, medium → `192k`, low → `160k` (uses `-abr 1`)

Path and filename templates
Placeholders supported in `file_path` and `filename_prefix`:
- `[time(%Y-%m-%d)]` → formatted time (strftime)
- `[date]` → `YYYY-MM-DD`
- `[datetime]` → `YYYY-MM-DD_HH-MM-SS`
- `[unix]` → epoch seconds
- `[guid]` / `[uuid]` → random UUID4 hex
- `[model]` → tries `extra_pnginfo` keys: `model`, `checkpoint`, `ckpt_name`, `model_name`; else `unknown`
- `[env(NAME)]` → environment variable `NAME`

Examples
- `audio/[time(%Y-%m-%d)]`
- `runs/[model]/[datetime]`
- `D:/Exports/[env(USERNAME)]/[guid]`

Notes
- On Windows, prefer `%H-%M-%S` instead of `%H:%M:%S` in strftime patterns.
- If neither ffmpeg nor lameenc is available, the node raises a clear error with install hints.

