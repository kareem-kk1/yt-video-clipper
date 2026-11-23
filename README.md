# yt-video-clipper


# 🎬 Local AI YouTube Shorts Generator

A local Python and Streamlit application that converts permitted YouTube videos into short, vertical videos using AI.

The application downloads a video using `yt-dlp`, transcribes its audio using OpenAI Whisper, identifies interesting moments using transcript-based scoring and optional local Ollama AI ranking, and renders vertical 9:16 clips using FFmpeg.

Generated videos include burned-in subtitles and downloadable `.srt` subtitle files. All processing and output generation happen locally on your computer.

---

## ✨ Features

- **YouTube downloading:** Download a single permitted YouTube video using `yt-dlp`.
- **AI transcription:** Generate timestamped transcripts using OpenAI Whisper.
- **Intelligent moment selection:** Identify interesting moments using transcript scoring.
- **Local LLM integration:** Optionally use Ollama with Llama 3.1 to rank and select engaging moments.
- **Multiple clip options:** Generate 1, 3, 5, or 7 clips.
- **Custom clip duration:** Choose 30, 45, or 60 seconds.
- **Vertical video rendering:** Convert landscape videos into 9:16 vertical videos at 1080 × 1920 resolution.
- **Blurred background:** Fill vertical video space using a blurred background when the original video is landscape.
- **Burned-in subtitles:** Generate subtitles in different visual styles.
- **Subtitle downloads:** Download individual `.srt` files.
- **NVIDIA GPU acceleration:** Use CUDA for Whisper transcription and NVIDIA NVENC for FFmpeg video encoding when supported.
- **Progress tracking:** Display download progress, processing stages, and clip-generation progress.
- **Session caching:** Reuse downloaded videos and transcripts when changing generation settings.
- **ZIP export:** Download all generated clips and subtitles in one ZIP archive.
- **Local processing:** Generated videos remain on your computer; the application does not automatically upload them to YouTube.

---

## 🛠️ Technology Stack

| Technology     | Purpose                             |
| -------------- | ----------------------------------- |
| Python         | Main programming language           |
| Streamlit      | User interface                      |
| yt-dlp         | YouTube video downloading           |
| OpenAI Whisper | Audio transcription                 |
| Ollama         | Local LLM-based moment selection    |
| Llama 3.1      | AI moment ranking                   |
| FFmpeg         | Video processing and rendering      |
| NVIDIA CUDA    | GPU acceleration for transcription  |
| NVIDIA NVENC   | Hardware-accelerated video encoding |
| JSON           | Transcript and metadata caching     |

---

## 💻 System Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.10 or 3.11 recommended.
- FFmpeg and ffprobe installed.
- Internet connection for downloading permitted YouTube videos.
- Ollama is optional.
- NVIDIA GPU and compatible CUDA-enabled PyTorch installation for GPU transcription.
- NVIDIA NVENC-compatible GPU and driver for hardware video encoding.

### Recommended laptop configuration

The following configuration was tested during development:

| Component                      | Configuration                      |
| ------------------------------ | ---------------------------------- |
| GPU                            | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM                       | 4 GB                               |
| NVIDIA Driver                  | 572.40                             |
| CUDA reported by NVIDIA driver | 12.8                               |
| FFmpeg                         | 7.1.1 Full Build                   |
| Video encoder                  | h264_nvenc                         |
| Python                         | 3.11 recommended                   |

The FFmpeg 7.1.1 configuration successfully passed an NVIDIA NVENC encoding test on this system.

Your hardware and driver requirements may differ.

---

# 🚀 Installation Guide

## 1. Clone or download the project

Open Command Prompt or PowerShell in your project directory.

Example:

```bat
cd C:\Users\KAREEM\OneDrive\Desktop\youtube_shorts_generator
```

## 2. Create a virtual environment

```bat
py -3.11 -m venv .venv
```

Activate it:

```bat
.venv\Scripts\activate
```

Upgrade pip:

```bat
python -m pip install --upgrade pip
```

## 3. Install Python dependencies

```bat
pip install -r requirements.txt
```

If you use CUDA, verify that your installed PyTorch build supports it.

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

Expected output on a correctly configured NVIDIA system:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

If CUDA is unavailable, check your NVIDIA driver and install a compatible CUDA-enabled PyTorch build.

Official PyTorch installation guide:

https://pytorch.org/get-started/locally/

**Important:** You generally do not need to install the full CUDA Toolkit just to use a compatible PyTorch CUDA wheel. Follow the PyTorch installation selector for the appropriate runtime package.

---

# 🎥 FFmpeg Installation and Configuration

FFmpeg is required for video rendering, subtitle burning, audio processing, and vertical video generation.

It also provides NVIDIA NVENC hardware encoding when a compatible build, GPU, and driver are available.

## 1. Why FFmpeg 7.1.1 is recommended for this setup

During development, FFmpeg 9.0.2 was installed through WinGet.

However, its NVIDIA encoder returned this compatibility error:

```text
Driver does not support the required nvenc API version.
Required: 13.1
Found: 13.0
```

The installed NVIDIA driver was version 572.40.

FFmpeg 7.1.1 was subsequently installed and tested successfully with the NVIDIA RTX 3050 Laptop GPU.

For this particular setup, use the tested FFmpeg 7.1.1 build rather than assuming the newest FFmpeg release will work with the installed NVIDIA driver.

## 2. Download FFmpeg 7.1.1

Download a Windows FFmpeg 7.1.1 Full Build from:

https://www.gyan.dev/ffmpeg/builds/

Extract the archive into:

```text
C:\ffmpeg\ffmpeg-7.1.1-full_build\
```

Your executable paths should be:

```text
C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe
C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffprobe.exe
```

## 3. Add FFmpeg to Windows PATH

Add this directory to your Windows environment PATH:

```text
C:\ffmpeg\ffmpeg-7.1.1-full_build\bin
```

Then close and reopen Command Prompt.

Verify FFmpeg:

```bat
ffmpeg -version
```

Verify ffprobe:

```bat
ffprobe -version
```

Both commands should execute successfully.

## 4. Verify NVIDIA NVENC support

Check whether your FFmpeg build includes the NVIDIA encoder:

```bat
ffmpeg -encoders | findstr nvenc
```

You should see an encoder named:

```text
h264_nvenc
```

Test actual GPU encoding:

```powershell
ffmpeg -f lavfi -i "testsrc=duration=5:size=1920x1080:rate=30" -pix_fmt yuv420p -c:v h264_nvenc -preset p4 -rc vbr -cq 23 -b:v 0 -f null -
```

A successful test should finish without an NVENC driver compatibility error.

### Configure the Python application

The video-rendering module should use the FFmpeg 7.1.1 executable directly.

In `clip_window.py`:

```python
from pathlib import Path

FFMPEG_PATH = Path(
    r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
)
```

The FFmpeg subprocess command should use:

```python
str(FFMPEG_PATH)
```

rather than relying on an ambiguous FFmpeg executable elsewhere on PATH.

For NVIDIA hardware encoding, the rendering command should use:

```python
"-c:v", "h264_nvenc",
"-preset", "p4",
"-rc", "vbr",
"-cq", "23",
"-b:v", "0",
"-pix_fmt", "yuv420p",
```

Audio encoding:

```python
"-c:a", "aac",
"-b:a", "128k",
"-movflags", "+faststart",
```

**Note:** The application must explicitly use `h264_nvenc` to take advantage of NVIDIA hardware encoding. Merely installing an NVENC-capable FFmpeg build does not automatically enable GPU encoding.

The application's FFmpeg availability check must also point to the configured executable or use the updated PATH.

---

# ⚡ CUDA and GPU Configuration

The application supports three processing-device options:

| Device | Description                                |
| ------ | ------------------------------------------ |
| CUDA   | Explicitly use the NVIDIA GPU              |
| Auto   | Use CUDA when available; otherwise use CPU |
| CPU    | Perform transcription using the processor  |

The default device is CUDA in the current application configuration.

## Verify NVIDIA GPU

Run:

```bat
nvidia-smi
```

This displays:

- GPU model
- NVIDIA driver version
- GPU memory usage
- CUDA version reported by the driver
- Running GPU processes

## Verify PyTorch CUDA

```bat
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is:

```text
True
```

PyTorch can access CUDA.

Check GPU memory:

```bat
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')"
```

## Understanding GPU acceleration

The application has two separate GPU-acceleration stages:

**1. Whisper transcription**

CUDA accelerates model inference when the installed PyTorch build and runtime support the GPU.

**2. FFmpeg video rendering**

NVIDIA NVENC accelerates H.264 video encoding using the GPU's dedicated video encoder.

YouTube downloading through `yt-dlp` is network-bound and does not become faster simply by selecting CUDA.

---

# 🤖 Optional: Install and Configure Ollama

Ollama provides local LLM inference for ranking interesting transcript moments.

Download Ollama:

https://ollama.com/download

Install it and open a terminal.

Pull the configured model:

```bat
ollama pull llama3.1:latest
```

Check available models:

```bat
ollama list
```

Start the Ollama service if it is not already running:

```bat
ollama serve
```

Keep Ollama running while using the application.

The Streamlit sidebar includes:

- Use Ollama for AI moment selection
- Ollama model

Default model:

```text
llama3.1:latest
```

If Ollama is unavailable or returns an invalid response, the selection module should use its transcript-based fallback when that fallback is configured.

Ollama model inference can consume additional system RAM and GPU memory. On a laptop with 4 GB VRAM, avoid running multiple GPU-heavy tasks simultaneously.

---

# ▶️ Run the Application

Activate the virtual environment:

```bat
.venv\Scripts\activate
```

Start Streamlit:

```bat
streamlit run application.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

# ⚙️ Application Settings

The sidebar provides the following generation options:

| Setting           | Available options                     |
| ----------------- | ------------------------------------- |
| Number of clips   | 1, 3, 5, 7                            |
| Clip duration     | 30, 45, 60 seconds                    |
| Whisper model     | tiny, base, small, medium             |
| Processing device | CUDA, Auto, CPU                       |
| Subtitle style    | Bold white, Yellow highlight, Minimal |
| Ollama ranking    | Enabled / Disabled                    |
| Ollama model      | llama3.1:latest by default            |

## Recommended settings for a 4 GB VRAM laptop

| Setting           | Recommendation   |
| ----------------- | ---------------- |
| Whisper model     | base             |
| Processing device | CUDA             |
| Clip count        | 3                |
| Clip duration     | 30 or 45 seconds |
| Subtitle style    | Bold white       |
| Ollama            | Optional         |

Use the small Whisper model if you need better transcription quality and have sufficient available memory.

The medium model can require significantly more memory and processing time.

If CUDA initialization fails, select CPU or Auto, depending on the behavior of the installed transcription module.

---

# 📊 Processing Workflow

The application processes videos through the following stages:

### Step 1: Download video

The application uses `yt-dlp` to download the supplied YouTube video.

Download progress is displayed in the Streamlit interface.

### Step 2: Transcribe audio

OpenAI Whisper extracts speech and generates timestamped transcript segments.

The transcript is saved locally for reuse.

### Step 3: Generate candidate moments

The application analyzes transcript segments and creates candidate time windows based on the requested clip duration.

### Step 4: Select interesting moments

Candidate windows are ranked using transcript-based scoring.

When enabled, Ollama helps select distinct and engaging moments.

### Step 5: Render vertical clips

FFmpeg processes the selected video segments and creates vertical 9:16 videos.

The rendering process includes:

- Video trimming
- Vertical formatting
- Blurred background for landscape footage
- Subtitle generation
- Subtitle burning
- H.264 encoding
- AAC audio encoding

### Step 6: Create ZIP bundle

The generated MP4 and subtitle files are packaged into a ZIP archive.

---

# ♻️ Caching and Performance

The application uses Streamlit session state to avoid repeating expensive processing.

### Video caching

When the same YouTube URL is submitted again, the application can reuse the downloaded video if its cached file is still available.

### Transcript caching

The transcript is reused when the video, Whisper model, and requested device configuration match the cached configuration.

Changing the Whisper model or processing device triggers transcription again.

### Generation settings

Changing the following settings does not require downloading or transcribing the video again:

- Clip count
- Clip duration
- Subtitle style
- Ollama selection
- Ollama model

The application can reuse the cached video and transcript and repeat moment selection and rendering.

### Cache lifetime

The cache is maintained within the current Streamlit browser session.

It is not a permanent database and should not be assumed to survive a new session or application restart.

Use the **Clear cached video and transcript** option in the sidebar to remove the active cached job.

---

# 📁 Project Structure

A typical project structure:

```text
youtube_shorts_generator/
│
├── application.py
├── yt_downloader.py
├── transcript.py
├── llama_call.py
├── clip_window.py
├── logger_config.py
│
├── requirements.txt
├── README.md
│
├── .venv/
│
├── outputs/
│   └── DD_MM_YYYY_HH_MM_SS_clips3_time45/
│       ├── short_01.mp4
│       ├── short_01.ass
│       ├── short_01.srt
│       ├── short_02.mp4
│       ├── short_02.ass
│       ├── short_02.srt
│       └── shorts_bundle.zip
│
├── work/
│   └── <job-id>/
│       ├── downloaded_video.mp4
│       ├── transcript.json
│       └── transcript_metadata.json
│
└── logs/
    └── application.log
```

Actual downloaded video filenames and log filenames may differ depending on the downloader and logging configuration.

Generated output folders use a timestamp-based naming convention.

Example:

```text
21_11_2026_15_30_00_clips3_time45
```

---

# 📦 Generated Output

Each completed generation job can contain:

| File                  | Description                                       |
| --------------------- | ------------------------------------------------- |
| `short_01.mp4`      | First generated vertical clip                     |
| `short_01.ass`      | Advanced SubStation Alpha subtitle file           |
| `short_01.srt`      | Standard subtitle file                            |
| `short_02.mp4`      | Second generated vertical clip                    |
| `shorts_bundle.zip` | ZIP containing generated clips and subtitle files |

The Streamlit interface provides individual video downloads, subtitle downloads, and a ZIP download.

---

# 🔧 Troubleshooting

## 1. FFmpeg not found

Verify:

```bat
ffmpeg -version
ffprobe -version
```

If either command fails:

- Check that FFmpeg is extracted correctly.
- Add the FFmpeg `bin` directory to PATH.
- Restart the terminal.
- Verify that the application uses the intended FFmpeg executable.

## 2. NVENC driver compatibility error

Example:

```text
Driver does not support the required nvenc API version.
```

Possible causes:

- FFmpeg is too new for the installed NVIDIA driver.
- The selected FFmpeg build does not include NVENC.
- The NVIDIA driver is outdated or incompatible.

For the RTX 3050 and driver 572.40 configuration tested during development, FFmpeg 7.1.1 successfully encoded using `h264_nvenc`.

Verify the encoder:

```bat
ffmpeg -encoders | findstr nvenc
```

Then run the NVENC test from the FFmpeg installation section.

If hardware encoding is unavailable, use a compatible FFmpeg build and configure a software encoder such as `libx264` if the application supports it.

## 3. CUDA unavailable

Check:

```bat
nvidia-smi
```

Then:

```bat
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable:

- Verify the NVIDIA driver.
- Check that the active virtual environment has CUDA-enabled PyTorch.
- Verify PyTorch and CUDA runtime compatibility.
- Restart the terminal after changing dependencies.
- Select CPU or Auto in the application if necessary.

## 4. YouTube download fails

Possible reasons:

- Video is private.
- Video is age-restricted.
- Video is region-restricted.
- Video requires membership.
- Video is unavailable.
- YouTube or yt-dlp has changed its extraction behavior.

Only download videos you own or have permission to download and edit.

Do not attempt to bypass access controls.

## 5. Ollama connection error

Verify that Ollama is running:

```bat
ollama list
```

Check that the configured model exists:

```bat
ollama run llama3.1:latest
```

Alternatively, disable Ollama moment selection in the Streamlit sidebar.

## 6. Whisper runs out of GPU memory

Try:

- Selecting the `base` model.
- Closing other GPU-intensive applications.
- Reducing the number of simultaneous GPU workloads.
- Selecting CPU if GPU memory remains insufficient.

## 7. No suitable moments found

Try:

- Selecting a shorter clip duration.
- Reducing the number of requested clips.
- Checking whether the transcript contains enough meaningful speech.
- Disabling Ollama to test transcript-based selection.

## 8. Download progress or processing appears stuck

Video downloading depends on network speed and the selected YouTube format.

Whisper transcription does not provide a reliable percentage through the standard transcription call used in this application. The interface therefore displays a processing spinner and advances the overall progress when transcription completes rather than inventing a transcription percentage.

---

# ⚖️ Copyright and Responsible Use

This project is intended for personal, educational, and authorized content-editing workflows.

- Only process videos you own or have permission to download and edit.
- Follow YouTube's Terms of Service.
- Respect copyright and licensing restrictions.
- Review all generated clips before publishing.
- Do not use the application to bypass access controls or download restricted content without authorization.

This application does not automatically publish videos to YouTube.

It creates local MP4 files for review and manual uploading.

---

# 🚧 Current Limitations

This project is an MVP, and generated clips may require manual review.

Known limitations include:

- Moment selection is primarily transcript-driven.
- Visual-only highlights may be missed.
- Jokes that depend on visual context may be selected incorrectly.
- Some clips may lack sufficient context at their beginning or end.
- Whisper transcription quality depends on audio quality and language.
- Ollama selection quality depends on the chosen model.
- GPU acceleration depends on compatible drivers, PyTorch, FFmpeg, and hardware.
- Download speed depends primarily on network conditions and video availability.

YouTube Shorts eligibility and presentation requirements can change. Check YouTube's current publishing requirements before uploading.

---

# 🔮 Future Improvements

Potential future enhancements:

- Scene detection and visual highlight recognition.
- AI-generated titles and descriptions.
- Automatic hook detection.
- Better sentence-boundary-aware clip selection.
- Word-level subtitle highlighting.
- Animated captions.
- Face detection and subject tracking.
- More subtitle templates.
- Automatic aspect-ratio optimization.
- Improved clip quality scoring.
- Persistent caching across application sessions.
- Batch processing for multiple videos.

---

# 👨‍💻 Development

Contributions, suggestions, and improvements are welcome.

If you find a bug or have an idea for improving the application, feel free to open an issue or submit a pull request.

**Built with Python, Streamlit, Whisper, Ollama, FFmpeg, and NVIDIA GPU acceleration**

# Local AI YouTube Shorts Generator

A local Streamlit application that downloads a permitted YouTube video, transcribes it with Faster-Whisper, ranks interesting transcript windows (optionally with Ollama), and renders vertical 9:16 MP4 clips with burned-in subtitles.

## Features

- YouTube URL input (single video, not playlists)
- Generate 3, 5, or 7 clips
- Clip duration: 30, 45, or 60 seconds
- Local transcription with Faster-Whisper
- Optional local Ollama ranking (default model: `llama3.1:8b`)
- 9:16 vertical video output with blurred background for landscape source footage
- Burned-in subtitles plus downloadable `.srt` files
- Individual MP4 downloads and a ZIP bundle

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- FFmpeg and ffprobe installed and available on PATH
- Ollama is optional. If enabled, Ollama must be running locally and the selected model must be pulled.
- Process only videos you own or have permission to download and edit. Follow YouTube's terms and copyright law.

## Windows installation

Open Command Prompt in this project folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install FFmpeg

Using winget:

```bat
winget install Gyan.FFmpeg
```

Close and reopen Command Prompt after installation, then verify:

```bat
ffmpeg -version
ffprobe -version
```

### Optional: configure Ollama

If Ollama is installed:

```bat
ollama pull llama3.1:8b
ollama serve
```

Keep Ollama running in another terminal. In the app, leave **Use Ollama to rank moments** enabled. If Ollama is unavailable or returns an invalid response, the app automatically falls back to transcript-based scoring.

## Run

```bat
streamlit run app.py
```

Open the local URL shown by Streamlit, usually http://localhost:8501.

## Recommended settings for a laptop with 4 GB VRAM

- Whisper model: `base` for speed, `small` for better transcription when memory permits.
- Device: `auto`; the app falls back to CPU if CUDA initialization fails.
- Start with 3 clips at 30 or 45 seconds.
- Avoid running other GPU-heavy applications while transcribing.

Faster-Whisper CUDA support depends on compatible NVIDIA drivers and CUDA/cuDNN runtime libraries. If CUDA fails, select CPU.

## Output

Generated files are saved under:

```text
outputs/<job-id>/
  short_01.mp4
  short_01.srt
  short_02.mp4
  short_02.srt
  ...
  selection.json
```

The app also offers a ZIP containing all clips and subtitles.

## How moment selection works

1. Whisper produces timestamped transcript segments.
2. The app builds fixed-length candidate windows.
3. It scores candidates using speech density, hook words, and sentence completion.
4. When enabled, Ollama selects distinct, self-contained moments from the highest-scoring candidates.
5. FFmpeg renders each selected window in 1080x1920 with subtitles.

This is an MVP: automated selection is transcript-driven and may miss visual-only highlights, jokes that depend on visuals, or context that spans longer than the selected window. Review each clip before publishing.

## Troubleshooting

- **FFmpeg not found:** install FFmpeg, add its `bin` directory to PATH, and restart the terminal.
- **YouTube download fails:** the video may be private, age-restricted, region-restricted, members-only, or otherwise unavailable to yt-dlp. Do not attempt to bypass access controls.
- **Ollama connection error:** start Ollama or turn off the Ollama ranking option.
- **CUDA error:** select CPU in the sidebar.
- **No suitable moments:** choose a shorter clip length or fewer clips.

## Notes

- This project does not upload videos to YouTube. It creates local MP4 files ready for review and manual upload.
- YouTube Shorts eligibility and presentation rules can change; check YouTube's current requirements before publishing.
