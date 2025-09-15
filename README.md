## Save MP3 node for ComfyUI

<img width="331" height="222" alt="image" src="https://github.com/user-attachments/assets/f5c17ac3-b3dc-4721-a133-1eaedc15a85a" />

I felt the comfy-core node lacked some functionality so I made one with...

- Audio input: accepts common formats used by audio nodes
- File path: choose ComfyUI output subfolder or absolute path on any drive
- Filename prefix: like the comfy-core node
- Bitrate mode: variable, constant, average (libmp3lame)
- Quality: low, medium, high (mapped per mode)
- Audio output for preview with external node
- Bitrate info output

...while keeping it as simple as possible. 
  
### Installation
1) Go to the `custom_nodes/` directory in ComfyUI.
2) Clone or copy the `comfyui-save-mp3`-folder
   ```bashcd
   git clone https://github.com/Dehypnotic/comfyui-save-mp3.git
3) Restart ComfyUI.

### Bitrate/quality mapping
- Variable (VBR): high→`-q:a 0`, medium→`-q:a 4`, low→`-q:a 7`
- Constant (CBR): high→`320k`, medium→`192k`, low→`128k`
- Average (ABR): high→`256k`, medium→`192k`, low→`160k` (uses `-abr 1`)

### Example

<img width="937" height="550" alt="image" src="https://github.com/user-attachments/assets/daa7ebcf-b623-43d6-a6a2-163fa9e6bb0f" />

### Notes
- Encoding uses lameenc (if installed) or ffmpeg. ffmpeg can be discovered via a bundled `bin/ffmpeg`, a system PATH, or auto-downloaded by `imageio-ffmpeg`.
- If neither lameenc nor any ffmpeg is found, the node raises an error with guidance.

