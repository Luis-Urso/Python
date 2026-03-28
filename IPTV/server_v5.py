"""
Corporate TV – Video + Audio Streaming Server (v5)
===================================================
Source Modes (switchable at runtime via API or dashboard):
  - playlist   : Plays video files from the media/ folder (default)
  - screen     : Captures the server's screen and broadcasts it live
  - livefile   : Streams an MKV/FLV/MP4 file being written in real-time
                 Uses FFmpeg pipe so it works even while the file is recording.
                 ** For OBS: set Recording Format to MKV or FLV (not MP4) **
  - streamurl  : Reads any live stream URL (RTMP, RTSP, UDP, HLS, etc.)
                 Use this with OBS Custom Output → rtmp://localhost:1935/live

NEW in v5 - Audio Streaming:
  - /audio endpoint streams audio as MP3 chunks
  - Dashboard includes synchronized audio player
  - All sources (playlist, livefile, streamurl) support audio
  - Screen capture can optionally capture system audio (requires additional setup)

Usage:
    pip install flask opencv-python-headless mss
    python server_v5.py

OBS Integration (two options):
  1. Record to MKV → use "livefile" source with the .mkv path
  2. Stream to RTMP → use "streamurl" source with the RTMP URL
     OBS Settings → Stream → Custom → rtmp://localhost:1935/live
     (requires an RTMP server like nginx-rtmp or mediamtx)

  Simplest OBS setup (no RTMP server needed):
     OBS → Settings → Output → Recording → Format: MKV
     Start recording, then POST /api/source/livefile {"file_path":"/path/to/file.mkv"}
"""

import os
import sys
import time
import json
import glob
import struct
import threading
import subprocess
import queue
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, send_file, request

try:
    import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5006
MEDIA_DIR = "./media"
FRAME_RATE = 30
JPEG_QUALITY = 85
SEEK_SECONDS = 10
SCREEN_CAPTURE_FPS = 30
SCREEN_MONITOR = 1
LIVE_FILE_POLL_INTERVAL = 0.5
PLAYLIST = []

# Audio settings
AUDIO_BITRATE = "128k"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_CHUNK_SIZE = 4096  # bytes per audio chunk

# =============================================================================
# APPLICATION SETUP
# =============================================================================
app = Flask(__name__)
Path(MEDIA_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# AUDIO STREAMER BASE CLASS
# =============================================================================
class AudioStreamer:
    """Base class for audio streaming using FFmpeg to produce MP3 chunks."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self._process = None
        self._thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.has_audio = False

    def _start_audio_process(self, input_args: list):
        """Start FFmpeg process to extract and encode audio."""
        try:
            cmd = (
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]
                + input_args
                + [
                    "-vn",  # No video
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    AUDIO_BITRATE,
                    "-ar",
                    str(AUDIO_SAMPLE_RATE),
                    "-ac",
                    str(AUDIO_CHANNELS),
                    "-f",
                    "mp3",
                    "pipe:1",
                ]
            )

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=AUDIO_CHUNK_SIZE * 4,
            )
            return True
        except Exception as e:
            print(f"[AUDIO] Failed to start FFmpeg: {e}")
            return False

    def _audio_read_loop(self):
        """Read audio data from FFmpeg and queue it."""
        while self.running and self._process:
            try:
                chunk = self._process.stdout.read(AUDIO_CHUNK_SIZE)
                if not chunk:
                    if self._process.poll() is not None:
                        break
                    continue

                self.has_audio = True
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    # Discard oldest chunk if queue is full
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(chunk)
                    except queue.Empty:
                        pass
            except Exception as e:
                print(f"[AUDIO] Read error: {e}")
                break

    def get_audio_chunk(self, timeout=0.1):
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_audio(self):
        """Stop the audio process."""
        self.running = False
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.has_audio = False


# =============================================================================
# SCREEN CAPTURE SOURCE
# =============================================================================
class ScreenCaptureSource:
    """Captures the server's screen using mss."""

    def __init__(self, monitor: int = 1, fps: int = 15):
        self.monitor = monitor
        self.fps = fps
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self._thread = None
        self.has_audio = False  # Screen capture doesn't have audio by default

    def start(self):
        if not MSS_AVAILABLE:
            print("[ERROR] Screen capture requires 'mss'. Install: pip install mss")
            return False
        if self.running:
            return True
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[SCREEN] Started capturing monitor {self.monitor} at {self.fps} FPS")
        return True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[SCREEN] Stopped")

    def _capture_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor]
            delay = 1.0 / self.fps
            while self.running:
                try:
                    img = sct.grab(monitor)
                    frame = np.array(img)[:, :, :3].copy()
                    with self.lock:
                        self.frame = frame
                except Exception as e:
                    print(f"[SCREEN] Capture error: {e}")
                time.sleep(delay)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_audio_chunk(self, timeout=0.1):
        """Screen capture has no audio by default."""
        return None


# =============================================================================
# LIVE FILE SOURCE (FFmpeg pipe — works with MKV/FLV while recording)
# =============================================================================
class LiveFileSource(AudioStreamer):
    """Reads a video file being written in real-time using FFmpeg.

    Unlike OpenCV's VideoCapture which needs the moov atom (MP4 only writes
    it at the end), FFmpeg can progressively demux MKV, FLV, and MPEG-TS
    containers while they're still being written.

    For OBS: set Recording Format to MKV or FLV in Settings → Output.

    Key fixes for live files:
      - No -re flag (that throttles reading and stalls on fresh files)
      - Low probesize/analyzeduration so FFmpeg starts immediately
      - +genpts flag to handle incomplete timestamps
      - Retries with increasing wait when file has no data yet
    """

    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.frame = None
        self._video_process = None
        self._video_thread = None
        self.width = 0
        self.height = 0
        self.fps = FRAME_RATE

    def start(self, file_path: str):
        self.stop()
        self.file_path = file_path
        self.running = True
        self._video_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._video_thread.start()
        # Start audio extraction in separate thread
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        print(f"[LIVE] Watching file via FFmpeg: {file_path}")
        return True

    def stop(self):
        self.running = False
        self.stop_audio()
        if self._video_process:
            try:
                self._video_process.kill()
                self._video_process.wait(timeout=3)
            except Exception:
                pass
            self._video_process = None
        if self._video_thread:
            self._video_thread.join(timeout=5)
        print("[LIVE] Stopped")

    def _wait_for_file(self):
        """Wait until the file exists and has some data (at least 1KB)."""
        print(f"[LIVE] Waiting for file to appear: {self.file_path}")
        for _ in range(120):  # Up to 60 seconds
            if not self.running:
                return False
            if (
                os.path.isfile(self.file_path)
                and os.path.getsize(self.file_path) > 1024
            ):
                return True
            time.sleep(0.5)
        print("[LIVE] Timed out waiting for file")
        return False

    def _probe_resolution(self):
        """Use ffprobe to get resolution. Retries because the file may still be initializing."""
        for attempt in range(40):  # Up to 20 seconds of retries
            if not self.running:
                return False
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-probesize",
                        "32768",  # Small probe — don't wait for lots of data
                        "-analyzeduration",
                        "500000",  # 0.5s max analysis
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,r_frame_rate",
                        "-of",
                        "json",
                        self.file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    self.width = int(streams[0].get("width", 1920))
                    self.height = int(streams[0].get("height", 1080))
                    # Parse fps from "30/1" format
                    fps_str = streams[0].get("r_frame_rate", "30/1")
                    parts = fps_str.split("/")
                    if len(parts) == 2 and int(parts[1]) > 0:
                        self.fps = int(parts[0]) / int(parts[1])
                    else:
                        self.fps = FRAME_RATE
                    print(
                        f"[LIVE] Detected: {self.width}x{self.height} @ {self.fps:.1f} FPS"
                    )
                    return True
            except (
                json.JSONDecodeError,
                KeyError,
                ValueError,
                subprocess.TimeoutExpired,
            ):
                pass
            time.sleep(0.5)

        # Fallback defaults
        if self.width == 0:
            self.width = 1920
            self.height = 1080
        print(
            f"[LIVE] Probe failed, using {self.width}x{self.height} @ {self.fps:.0f} FPS"
        )
        return True

    def _read_loop(self):
        """Spawn FFmpeg to decode the file and pipe raw BGR frames."""
        if not self._wait_for_file():
            return
        if not self._probe_resolution():
            return

        frame_size = self.width * self.height * 3

        while self.running:
            try:
                cmd = [
                    "ffmpeg",
                    # Input options — tuned for live/incomplete files
                    "-probesize",
                    "32768",  # Tiny probe size — start fast
                    "-analyzeduration",
                    "500000",  # 0.5s analysis max
                    "-fflags",
                    "+genpts+discardcorrupt+nobuffer",
                    "-flags",
                    "low_delay",
                    "-i",
                    self.file_path,
                    # Output options — raw video pipe
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{self.width}x{self.height}",
                    "-an",  # Drop audio (handled separately)
                    "-vsync",
                    "cfr",  # Constant frame rate output
                    "-v",
                    "error",
                    "pipe:1",
                ]

                print(f"[LIVE] Starting FFmpeg video decoder...")
                self._video_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_size * 2,
                )

                frames_read = 0
                frame_delay = 1.0 / self.fps

                while self.running:
                    raw = self._video_process.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        # Check if FFmpeg errored
                        if self._video_process.poll() is not None:
                            stderr = self._video_process.stderr.read().decode(
                                errors="ignore"
                            )
                            if stderr.strip():
                                print(f"[LIVE] FFmpeg error: {stderr.strip()[:200]}")
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    )
                    with self.lock:
                        self.frame = frame
                    frames_read += 1

                    if frames_read == 1:
                        print(f"[LIVE] First frame received! Streaming...")

                    # Pace output to match video FPS so we don't burn CPU
                    time.sleep(frame_delay)

                print(f"[LIVE] FFmpeg exited after {frames_read} frames")

            except Exception as e:
                print(f"[LIVE] Error: {e}")

            # Clean up
            if self._video_process:
                try:
                    self._video_process.kill()
                    self._video_process.wait(timeout=2)
                except Exception:
                    pass
                self._video_process = None

            if self.running:
                # If we got zero frames, the file probably isn't ready yet — wait longer
                wait = 1 if frames_read > 0 else 3
                print(f"[LIVE] Retrying in {wait}s...")
                time.sleep(wait)

    def _audio_loop(self):
        """Spawn FFmpeg to extract audio from the live file."""
        if not self._wait_for_file():
            return

        while self.running:
            try:
                input_args = [
                    "-probesize",
                    "32768",
                    "-analyzeduration",
                    "500000",
                    "-fflags",
                    "+genpts+discardcorrupt+nobuffer",
                    "-i",
                    self.file_path,
                ]

                if self._start_audio_process(input_args):
                    print("[LIVE] Audio extraction started")
                    self._audio_read_loop()
                    print("[LIVE] Audio stream ended")
            except Exception as e:
                print(f"[LIVE] Audio error: {e}")

            if self.running:
                time.sleep(2)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


# =============================================================================
# STREAM URL SOURCE (RTMP, RTSP, UDP, HLS, etc.)
# =============================================================================
class StreamURLSource(AudioStreamer):
    """Reads a live stream from any URL that OpenCV/FFmpeg can handle.

    Examples:
      - rtmp://localhost:1935/live/stream
      - rtsp://camera-ip:554/stream
      - udp://@:1234
      - http://server/stream.m3u8
    """

    def __init__(self):
        super().__init__()
        self.url = ""
        self.frame = None
        self._video_thread = None

    def start(self, url: str):
        self.stop()
        self.url = url
        self.running = True
        self._video_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._video_thread.start()
        # Start audio extraction
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        print(f"[STREAM] Connecting to: {url}")
        return True

    def stop(self):
        self.running = False
        self.stop_audio()
        if self._video_thread:
            self._video_thread.join(timeout=5)
        print("[STREAM] Stopped")

    def _read_loop(self):
        while self.running:
            try:
                # OpenCV uses FFmpeg backend, so it supports RTMP/RTSP/etc.
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    print(f"[STREAM] Cannot open {self.url}, retrying in 3s...")
                    time.sleep(3)
                    continue

                print(f"[STREAM] Connected to {self.url}")

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("[STREAM] Lost connection, reconnecting...")
                        break
                    with self.lock:
                        self.frame = frame

                cap.release()

            except Exception as e:
                print(f"[STREAM] Error: {e}")

            if self.running:
                time.sleep(3)

    def _audio_loop(self):
        """Extract audio from the stream URL."""
        while self.running:
            try:
                input_args = [
                    "-i",
                    self.url,
                ]

                if self._start_audio_process(input_args):
                    print("[STREAM] Audio extraction started")
                    self._audio_read_loop()
                    print("[STREAM] Audio stream ended")
            except Exception as e:
                print(f"[STREAM] Audio error: {e}")

            if self.running:
                time.sleep(3)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


# =============================================================================
# PLAYLIST AUDIO SOURCE
# =============================================================================
class PlaylistAudioSource(AudioStreamer):
    """Extracts audio from playlist videos."""

    def __init__(self):
        super().__init__()
        self.current_file = ""
        self.seek_position = 0

    def start(self, file_path: str, seek_seconds: float = 0):
        """Start audio extraction from a video file."""
        self.stop_audio()
        self.current_file = file_path
        self.seek_position = seek_seconds
        self.running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        return True

    def _audio_loop(self):
        """Extract audio from the playlist video."""
        while self.running:
            try:
                input_args = ["-i", self.current_file]
                if self.seek_position > 0:
                    input_args = ["-ss", str(self.seek_position)] + input_args

                if self._start_audio_process(input_args):
                    print(
                        f"[PLAYLIST AUDIO] Started: {os.path.basename(self.current_file)}"
                    )
                    self._audio_read_loop()
            except Exception as e:
                print(f"[PLAYLIST AUDIO] Error: {e}")

            # Don't retry in playlist mode - wait for next track
            break

    def sync_to_video(self, file_path: str, position_seconds: float):
        """Sync audio to the current video position."""
        if file_path != self.current_file or not self.running:
            self.stop_audio()
            self.running = True
            self.current_file = file_path
            self.seek_position = position_seconds
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()


# =============================================================================
# VIDEO STREAMER (all sources + switching)
# =============================================================================
class VideoStreamer:
    """Manages all sources and routes frames to the MJPEG stream."""

    STATE_PLAYING = "playing"
    STATE_PAUSED = "paused"
    STATE_STOPPED = "stopped"

    SOURCE_PLAYLIST = "playlist"
    SOURCE_SCREEN = "screen"
    SOURCE_LIVEFILE = "livefile"
    SOURCE_STREAMURL = "streamurl"

    def __init__(self, media_dir: str, playlist: list[str] = None):
        self.media_dir = media_dir
        self.playlist = playlist or []
        self.current_index = 0
        self.lock = threading.Lock()
        self.cap = None
        self.state = self.STATE_STOPPED
        self.last_frame = None
        self.video_fps = FRAME_RATE
        self.source = self.SOURCE_PLAYLIST

        self.screen_source = ScreenCaptureSource(SCREEN_MONITOR, SCREEN_CAPTURE_FPS)
        self.live_source = LiveFileSource()
        self.stream_source = StreamURLSource()
        self.playlist_audio = PlaylistAudioSource()

        self._build_playlist()

        # Audio sync tracking
        self._last_audio_sync_file = ""
        self._last_audio_sync_pos = 0

    def _build_playlist(self):
        if not self.playlist:
            extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm", "*.flv")
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(self.media_dir, ext)))
            self.playlist = sorted(files)
        else:
            self.playlist = [
                os.path.join(self.media_dir, f) if not os.path.isabs(f) else f
                for f in self.playlist
            ]
        if not self.playlist:
            print("[WARNING] No video files found in", self.media_dir)
        else:
            print(f"[INFO] Playlist loaded with {len(self.playlist)} videos:")
            for i, f in enumerate(self.playlist):
                print(f"  {i + 1}. {os.path.basename(f)}")

    def _open_video(self, index: int) -> bool:
        if not self.playlist:
            return False
        if self.cap is not None:
            self.cap.release()
        self.current_index = index % len(self.playlist)
        path = self.playlist[self.current_index]
        print(f"[PLAY] Now playing: {os.path.basename(path)}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open: {path}")
            return False
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or FRAME_RATE

        # Start audio for this video
        self.playlist_audio.start(path, 0)

        return True

    def _encode_frame(self, frame):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, jpeg = cv2.imencode(".jpg", frame, encode_params)
        return jpeg.tobytes()

    # -----------------------------------------------------------------
    # Source switching
    # -----------------------------------------------------------------
    def _stop_all_sources(self, except_source=None):
        """Stop all non-playlist sources except the one we're switching to."""
        if except_source != self.SOURCE_SCREEN:
            self.screen_source.stop()
        if except_source != self.SOURCE_LIVEFILE:
            self.live_source.stop()
        if except_source != self.SOURCE_STREAMURL:
            self.stream_source.stop()
        if except_source != self.SOURCE_PLAYLIST:
            self.playlist_audio.stop_audio()
        if self.cap is not None and except_source != self.SOURCE_PLAYLIST:
            self.cap.release()
            self.cap = None

    def switch_source(self, source: str, **kwargs):
        with self.lock:
            self._stop_all_sources(except_source=source)
            self.last_frame = None

            if source == self.SOURCE_PLAYLIST:
                self.source = self.SOURCE_PLAYLIST
                self.state = self.STATE_PLAYING
                self._open_video(self.current_index)
                print("[SOURCE] → Playlist")
                return True

            elif source == self.SOURCE_SCREEN:
                if not MSS_AVAILABLE:
                    print("[ERROR] 'mss' not installed. pip install mss")
                    return False
                self.source = self.SOURCE_SCREEN
                self.state = self.STATE_PLAYING

        # Start outside lock (spawns threads)
        if source == self.SOURCE_SCREEN:
            self.screen_source.start()
            print("[SOURCE] → Screen capture")
            return True

        if source == self.SOURCE_LIVEFILE:
            file_path = kwargs.get("file_path", "")
            if not file_path:
                print("[ERROR] No file_path provided")
                return False
            with self.lock:
                self.source = self.SOURCE_LIVEFILE
                self.state = self.STATE_PLAYING
            self.live_source.start(file_path)
            print(f"[SOURCE] → Live file: {file_path}")
            return True

        if source == self.SOURCE_STREAMURL:
            url = kwargs.get("url", "")
            if not url:
                print("[ERROR] No url provided")
                return False
            with self.lock:
                self.source = self.SOURCE_STREAMURL
                self.state = self.STATE_PLAYING
            self.stream_source.start(url)
            print(f"[SOURCE] → Stream URL: {url}")
            return True

        return False

    # -----------------------------------------------------------------
    # Audio chunk retrieval
    # -----------------------------------------------------------------
    def get_audio_chunk(self, timeout=0.1):
        """Get audio chunk from the current source."""
        with self.lock:
            source = self.source
            state = self.state

        if source == self.SOURCE_SCREEN:
            return self.screen_source.get_audio_chunk(timeout)
        elif source == self.SOURCE_LIVEFILE:
            return self.live_source.get_audio_chunk(timeout)
        elif source == self.SOURCE_STREAMURL:
            return self.stream_source.get_audio_chunk(timeout)
        elif source == self.SOURCE_PLAYLIST:
            if state == self.STATE_PLAYING:
                return self.playlist_audio.get_audio_chunk(timeout)
        return None

    # -----------------------------------------------------------------
    # Frame retrieval
    # -----------------------------------------------------------------
    def get_frame(self):
        with self.lock:
            # --- Screen ---
            if self.source == self.SOURCE_SCREEN:
                frame = self.screen_source.get_frame()
                if frame is not None:
                    self.last_frame = frame
                    return self._encode_frame(frame)
                return (
                    self._encode_frame(self.last_frame)
                    if self.last_frame is not None
                    else None
                )

            # --- Live file ---
            if self.source == self.SOURCE_LIVEFILE:
                frame = self.live_source.get_frame()
                if frame is not None:
                    self.last_frame = frame
                    return self._encode_frame(frame)
                return (
                    self._encode_frame(self.last_frame)
                    if self.last_frame is not None
                    else None
                )

            # --- Stream URL ---
            if self.source == self.SOURCE_STREAMURL:
                frame = self.stream_source.get_frame()
                if frame is not None:
                    self.last_frame = frame
                    return self._encode_frame(frame)
                return (
                    self._encode_frame(self.last_frame)
                    if self.last_frame is not None
                    else None
                )

            # --- Playlist ---
            if self.state == self.STATE_STOPPED or self.state == self.STATE_PAUSED:
                if self.last_frame is not None:
                    return self._encode_frame(self.last_frame)
                return None

            if self.cap is None or not self.cap.isOpened():
                if not self._open_video(self.current_index):
                    return None

            ret, frame = self.cap.read()
            if not ret:
                next_idx = (self.current_index + 1) % len(self.playlist)
                if not self._open_video(next_idx):
                    return None
                ret, frame = self.cap.read()
                if not ret:
                    return None

            self.last_frame = frame
            return self._encode_frame(frame)

    # -----------------------------------------------------------------
    # Playback controls (playlist mode only)
    # -----------------------------------------------------------------
    def play(self):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return
            if self.state == self.STATE_STOPPED:
                self._open_video(self.current_index)
            self.state = self.STATE_PLAYING
            # Resume audio
            if self.playlist and self.cap:
                pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / (self.video_fps or 30)
                self.playlist_audio.sync_to_video(
                    self.playlist[self.current_index], pos
                )

    def pause(self):
        with self.lock:
            if self.source == self.SOURCE_PLAYLIST and self.state == self.STATE_PLAYING:
                self.state = self.STATE_PAUSED
                self.playlist_audio.stop_audio()

    def toggle_pause(self):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return
            if self.state == self.STATE_PLAYING:
                self.state = self.STATE_PAUSED
                self.playlist_audio.stop_audio()
            elif self.state == self.STATE_PAUSED:
                self.state = self.STATE_PLAYING
                if self.playlist and self.cap:
                    pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / (self.video_fps or 30)
                    self.playlist_audio.sync_to_video(
                        self.playlist[self.current_index], pos
                    )
            else:
                self._open_video(self.current_index)
                self.state = self.STATE_PLAYING

    def stop(self):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return
            self.state = self.STATE_STOPPED
            self.playlist_audio.stop_audio()
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def skip_next(self):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return
            self._open_video((self.current_index + 1) % len(self.playlist))
            if self.state == self.STATE_STOPPED:
                self.state = self.STATE_PLAYING

    def skip_previous(self):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return
            self._open_video((self.current_index - 1) % len(self.playlist))
            if self.state == self.STATE_STOPPED:
                self.state = self.STATE_PLAYING

    def jump_to(self, index: int):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST:
                return False
            if 0 <= index < len(self.playlist):
                self._open_video(index)
                if self.state == self.STATE_STOPPED:
                    self.state = self.STATE_PLAYING
                return True
            return False

    def rewind(self, seconds: float = None):
        seconds = seconds or SEEK_SECONDS
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST or self.cap is None:
                return
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            target = max(0, current - (self.video_fps * seconds))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            if self.state in (self.STATE_PAUSED, self.STATE_STOPPED):
                ret, f = self.cap.read()
                if ret:
                    self.last_frame = f
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            # Resync audio
            if self.playlist and self.state == self.STATE_PLAYING:
                pos = target / (self.video_fps or 30)
                self.playlist_audio.sync_to_video(
                    self.playlist[self.current_index], pos
                )

    def fast_forward(self, seconds: float = None):
        seconds = seconds or SEEK_SECONDS
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST or self.cap is None:
                return
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            target = min(total - 1, current + (self.video_fps * seconds))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            if self.state in (self.STATE_PAUSED, self.STATE_STOPPED):
                ret, f = self.cap.read()
                if ret:
                    self.last_frame = f
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            # Resync audio
            if self.playlist and self.state == self.STATE_PLAYING:
                pos = target / (self.video_fps or 30)
                self.playlist_audio.sync_to_video(
                    self.playlist[self.current_index], pos
                )

    def seek_to(self, pct: float):
        with self.lock:
            if self.source != self.SOURCE_PLAYLIST or self.cap is None:
                return
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            target = max(0, min(int(total - 1), int((pct / 100.0) * total)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            if self.state in (self.STATE_PAUSED, self.STATE_STOPPED):
                ret, f = self.cap.read()
                if ret:
                    self.last_frame = f
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            # Resync audio
            if self.playlist and self.state == self.STATE_PLAYING:
                pos = target / (self.video_fps or 30)
                self.playlist_audio.sync_to_video(
                    self.playlist[self.current_index], pos
                )

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------
    def get_status(self):
        current_file = (
            os.path.basename(self.playlist[self.current_index])
            if self.playlist
            else "No media"
        )
        position_sec = duration_sec = progress_percent = 0.0
        has_audio = False

        if self.source == self.SOURCE_PLAYLIST and self.cap and self.cap.isOpened():
            fps = self.video_fps or 30
            cur_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            tot_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            position_sec = cur_f / fps
            duration_sec = tot_f / fps
            progress_percent = (cur_f / tot_f * 100) if tot_f > 0 else 0
            has_audio = self.playlist_audio.has_audio

        if self.source == self.SOURCE_SCREEN:
            current_file = f"Screen Capture (Monitor {SCREEN_MONITOR})"
            has_audio = False
        elif self.source == self.SOURCE_LIVEFILE:
            current_file = "Live File: " + (
                os.path.basename(self.live_source.file_path) or "None"
            )
            has_audio = self.live_source.has_audio
        elif self.source == self.SOURCE_STREAMURL:
            current_file = "Stream: " + (self.stream_source.url or "None")
            has_audio = self.stream_source.has_audio

        return {
            "source": self.source,
            "current_video": current_file,
            "playlist_position": self.current_index + 1,
            "playlist_total": len(self.playlist),
            "state": self.state if self.source == self.SOURCE_PLAYLIST else "playing",
            "position_seconds": round(position_sec, 1),
            "duration_seconds": round(duration_sec, 1),
            "progress_percent": round(progress_percent, 1),
            "server_time": datetime.now().isoformat(),
            "mss_available": MSS_AVAILABLE,
            "has_audio": has_audio,
        }


# Initialize
streamer = VideoStreamer(MEDIA_DIR, PLAYLIST if PLAYLIST else None)
streamer.play()


# =============================================================================
# ROUTES
# =============================================================================
def generate_mjpeg():
    frame_delay = 1.0 / FRAME_RATE
    while True:
        frame = streamer.get_frame()
        if frame is None:
            time.sleep(1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(frame_delay)


def generate_audio():
    """Generate MP3 audio stream."""
    while True:
        chunk = streamer.get_audio_chunk(timeout=0.5)
        if chunk:
            yield chunk
        else:
            # Send silence frame to keep connection alive
            time.sleep(0.05)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/stream")
def video_stream():
    return Response(
        generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/audio")
def audio_stream():
    """Stream audio as MP3."""
    return Response(
        generate_audio(),
        mimetype="audio/mpeg",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/status")
def api_status():
    return jsonify(streamer.get_status())


# --- Source ---
@app.route("/api/source/playlist", methods=["POST"])
def api_src_playlist():
    return jsonify(
        {"success": streamer.switch_source("playlist"), **streamer.get_status()}
    )


@app.route("/api/source/screen", methods=["POST"])
def api_src_screen():
    return jsonify(
        {"success": streamer.switch_source("screen"), **streamer.get_status()}
    )


@app.route("/api/source/livefile", methods=["POST"])
def api_src_livefile():
    data = request.get_json(silent=True) or {}
    return jsonify(
        {
            "success": streamer.switch_source(
                "livefile", file_path=data.get("file_path", "")
            ),
            **streamer.get_status(),
        }
    )


@app.route("/api/source/streamurl", methods=["POST"])
def api_src_streamurl():
    data = request.get_json(silent=True) or {}
    return jsonify(
        {
            "success": streamer.switch_source("streamurl", url=data.get("url", "")),
            **streamer.get_status(),
        }
    )


# --- Playback ---
@app.route("/api/play", methods=["POST"])
def api_play():
    streamer.play()
    return jsonify(streamer.get_status())


@app.route("/api/pause", methods=["POST"])
def api_pause():
    streamer.pause()
    return jsonify(streamer.get_status())


@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    streamer.toggle_pause()
    return jsonify(streamer.get_status())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    streamer.stop()
    return jsonify(streamer.get_status())


# --- Navigation ---
@app.route("/api/next", methods=["POST"])
def api_next():
    streamer.skip_next()
    return jsonify(streamer.get_status())


@app.route("/api/previous", methods=["POST"])
def api_prev():
    streamer.skip_previous()
    return jsonify(streamer.get_status())


@app.route("/api/jump", methods=["POST"])
def api_jump():
    data = request.get_json(silent=True) or {}
    idx = data.get("index", 0)
    streamer.jump_to(idx)
    return jsonify(streamer.get_status())


# --- Seek ---
@app.route("/api/rewind", methods=["POST"])
def api_rewind():
    data = request.get_json(silent=True) or {}
    seconds = data.get("seconds")
    streamer.rewind(seconds)
    return jsonify(streamer.get_status())


@app.route("/api/forward", methods=["POST"])
def api_forward():
    data = request.get_json(silent=True) or {}
    seconds = data.get("seconds")
    streamer.fast_forward(seconds)
    return jsonify(streamer.get_status())


@app.route("/api/seek", methods=["POST"])
def api_seek():
    data = request.get_json(silent=True) or {}
    pct = data.get("position", 0)
    streamer.seek_to(pct)
    return jsonify(streamer.get_status())


@app.route("/api/playlist")
def api_playlist():
    files = [os.path.basename(f) for f in streamer.playlist]
    return jsonify({"playlist": files, "current_index": streamer.current_index})


# =============================================================================
# DASHBOARD HTML
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corporate TV Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0f1a;color:#e2e8f0;min-height:100vh;padding:1rem}
.header{display:flex;align-items:center;gap:.7rem;margin-bottom:1rem;padding-bottom:.7rem;border-bottom:1px solid #1e3a5f}
.header h1{font-size:1.15rem;font-weight:600;color:#5eead4}
.dot{width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981}
.grid{display:grid;grid-template-columns:1fr 320px;gap:1rem}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:1rem}
.card h2{font-size:.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.6rem}
.preview{border-radius:8px;overflow:hidden;background:#000;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
.preview img{width:100%;height:100%;object-fit:contain}
.stat{font-size:1rem;font-weight:600;color:#e2e8f0;word-break:break-all}
.stat-label{color:#64748b;font-size:.75rem;margin-top:.15rem}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:20px;font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-top:.4rem}
.state-playing{background:#064e3b;color:#10b981}.state-paused{background:#78350f;color:#f59e0b}.state-stopped{background:#7f1d1d;color:#ef4444}
.source-bar{display:flex;gap:.35rem;margin-bottom:1rem;flex-wrap:wrap}
.src-btn{background:#1e3a5f;color:#94a3b8;border:1px solid #2d4a6f;padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.75rem;font-weight:600;transition:all .15s}
.src-btn:hover{background:#2d4a6f;color:#e2e8f0}.src-btn.active{background:#0d9488;color:#fff;border-color:#14b8a6}
.input-row{display:none;margin-top:.5rem;gap:.35rem}.input-row.show{display:flex}
.input-row input{flex:1;background:#0f1b2d;border:1px solid #2d4a6f;color:#e2e8f0;padding:.35rem .5rem;border-radius:6px;font-size:.78rem;font-family:monospace}
.input-row button{background:#0d9488;color:#fff;border:none;padding:.35rem .7rem;border-radius:6px;cursor:pointer;font-size:.75rem;font-weight:600}
.transport{display:flex;align-items:center;justify-content:center;gap:.35rem;margin-top:.9rem;flex-wrap:wrap}
.transport button{background:#1e3a5f;color:#e2e8f0;border:1px solid #2d4a6f;width:40px;height:40px;border-radius:10px;cursor:pointer;font-size:1rem;display:flex;align-items:center;justify-content:center;transition:background .15s}
.transport button:hover{background:#0d9488;border-color:#14b8a6}
.transport button.primary{width:48px;height:48px;background:#0d9488;border-color:#14b8a6;font-size:1.15rem}
.transport button.primary:hover{background:#14b8a6}
.transport button.stop-btn:hover{background:#dc2626;border-color:#ef4444}
.pbar-wrap{margin-top:.7rem}.pbar{width:100%;height:7px;background:#1e3a5f;border-radius:4px;cursor:pointer;overflow:hidden}
.pbar-fill{height:100%;background:#0d9488;border-radius:4px;transition:width .3s}
.pbar-time{display:flex;justify-content:space-between;font-size:.7rem;color:#64748b;margin-top:.2rem}
.pl-item{padding:.4rem .5rem;border-bottom:1px solid #1e3a5f;font-size:.8rem;display:flex;justify-content:space-between;align-items:center;cursor:pointer;border-radius:6px;transition:background .15s}
.pl-item:hover{background:#1e3a5f}.pl-item.active{color:#5eead4;font-weight:600;background:#0d3b4f}
.ep{font-family:monospace;font-size:.72rem;color:#94a3b8;line-height:1.75}
.ep code{background:#0f1b2d;padding:.1rem .3rem;border-radius:4px;color:#5eead4}
.src-lbl{display:inline-block;padding:.12rem .45rem;border-radius:12px;font-size:.6rem;font-weight:600;text-transform:uppercase;background:#1e3a5f;color:#5eead4;margin-left:.5rem}
.hint{font-size:.7rem;color:#64748b;margin-top:.3rem;line-height:1.4;padding:.4rem;background:#0f1b2d;border-radius:6px}
.audio-section{margin-top:1rem;padding:.7rem;background:#0f1b2d;border-radius:8px}
.audio-section h3{font-size:.75rem;color:#94a3b8;margin-bottom:.5rem;display:flex;align-items:center;gap:.4rem}
.audio-controls{display:flex;align-items:center;gap:.6rem}
.audio-btn{background:#1e3a5f;color:#e2e8f0;border:1px solid #2d4a6f;width:36px;height:36px;border-radius:8px;cursor:pointer;font-size:.9rem;display:flex;align-items:center;justify-content:center;transition:all .15s}
.audio-btn:hover{background:#0d9488;border-color:#14b8a6}
.audio-btn.active{background:#0d9488;border-color:#14b8a6}
.audio-btn.muted{background:#7f1d1d;border-color:#ef4444}
.volume-slider{flex:1;height:6px;-webkit-appearance:none;appearance:none;background:#1e3a5f;border-radius:3px;outline:none}
.volume-slider::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#0d9488;cursor:pointer}
.volume-slider::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#0d9488;cursor:pointer;border:none}
.audio-indicator{display:flex;align-items:center;gap:.3rem;font-size:.7rem;color:#64748b}
.audio-indicator.active{color:#10b981}
.audio-dot{width:6px;height:6px;border-radius:50%;background:#64748b}
.audio-indicator.active .audio-dot{background:#10b981;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
</style>
</head>
<body>
<div class="header">
  <div class="dot" id="dot"></div>
  <h1>Corporate TV — Server Dashboard</h1>
  <span class="src-lbl" id="src-lbl">playlist</span>
</div>
<div class="grid">
  <div class="card">
    <h2>Live Preview</h2>
    <div class="preview"><img src="/stream" alt="Live Stream"></div>
    
    <div class="audio-section">
      <h3>
        <span>🔊 Audio</span>
        <span class="audio-indicator" id="audio-ind"><span class="audio-dot"></span>streaming</span>
      </h3>
      <div class="audio-controls">
        <button class="audio-btn" id="audio-toggle" onclick="toggleAudio()" title="Toggle Audio">🔇</button>
        <input type="range" class="volume-slider" id="volume" min="0" max="100" value="80" onchange="setVolume(this.value)">
        <span id="vol-pct" style="font-size:.7rem;color:#64748b;min-width:35px">80%</span>
      </div>
      <audio id="audio-player" style="display:none"></audio>
    </div>
  </div>
  <div class="card">
    <h2>Source</h2>
    <div class="source-bar">
      <button class="src-btn active" id="sb-playlist" onclick="switchSrc('playlist')">Playlist</button>
      <button class="src-btn" id="sb-screen" onclick="switchSrc('screen')">Screen</button>
      <button class="src-btn" id="sb-livefile" onclick="toggleInput('li')">Live File</button>
      <button class="src-btn" id="sb-streamurl" onclick="toggleInput('su')">Stream URL</button>
    </div>
    <div class="input-row" id="li-row">
      <input id="li-path" placeholder="/path/to/recording.mkv (use MKV or FLV, not MP4)">
      <button onclick="startLive()">Go</button>
    </div>
    <div class="input-row" id="su-row">
      <input id="su-url" placeholder="rtmp://localhost:1935/live/stream">
      <button onclick="startStream()">Go</button>
    </div>
    <div class="hint" id="hint" style="display:none"></div>
 
    <h2 style="margin-top:1rem">Now Playing</h2>
    <div class="stat" id="cur-vid">Loading...</div>
    <div class="stat-label" id="pos-lbl"></div>
    <div id="st-badge" class="badge state-stopped">stopped</div>
 
    <div class="pbar-wrap" id="prog-sec">
      <div class="pbar" id="pbar" onclick="seekBar(event)"><div class="pbar-fill" id="pfill" style="width:0%"></div></div>
      <div class="pbar-time"><span id="t-cur">0:00</span><span id="t-tot">0:00</span></div>
    </div>
    <div class="transport" id="tp-sec">
      <button onclick="api('previous')" title="Previous">⏮</button>
      <button onclick="api('rewind')" title="Rewind 10s">⏪</button>
      <button onclick="api('stop')" class="stop-btn" title="Stop">⏹</button>
      <button onclick="api('toggle')" class="primary" id="pp-btn" title="Play/Pause">▶</button>
      <button onclick="api('forward')" title="Forward 10s">⏩</button>
      <button onclick="api('next')" title="Next">⏭</button>
    </div>
 
    <h2 style="margin-top:1rem">Playlist</h2>
    <div id="pl"></div>
  </div>
  <div class="card" style="grid-column:span 2">
    <h2>API Reference</h2>
    <div class="ep">
      <p>Video Stream: <code>GET /stream</code> (MJPEG)</p>
      <p>Audio Stream: <code>GET /audio</code> (MP3)</p>
      <p>Source: <code>POST /api/source/playlist</code> · <code>/screen</code> · <code>/livefile</code> {file_path} · <code>/streamurl</code> {url}</p>
      <p>Controls: <code>POST /api/play</code> · <code>/pause</code> · <code>/toggle</code> · <code>/stop</code></p>
      <p>Navigate: <code>POST /api/next</code> · <code>/previous</code> · <code>/jump</code> {index}</p>
      <p>Seek: <code>POST /api/rewind</code> · <code>/forward</code> {seconds} · <code>/seek</code> {position}</p>
    </div>
  </div>
</div>
<script>
let cSrc='playlist';
let audioEnabled=false;
let audioPlayer=null;
 
function fmt(s){if(!s||s<0)return'0:00';return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0')}
 
async function api(a,b){
  const o={method:'POST',headers:{'Content-Type':'application/json'}};
  if(b)o.body=JSON.stringify(b);
  try{await fetch('/api/'+a,o);upd();pl()}catch(e){}
}
 
function seekBar(e){
  const r=document.getElementById('pbar').getBoundingClientRect();
  api('seek',{position:Math.max(0,Math.min(100,((e.clientX-r.left)/r.width)*100))});
  // Restart audio to resync
  if(audioEnabled){restartAudio()}
}
 
function jumpTo(i){api('jump',{index:i});if(audioEnabled){setTimeout(restartAudio,500)}}
 
async function switchSrc(s){
  hideInputs();
  await api('source/'+s);
  if(audioEnabled){setTimeout(restartAudio,1000)}
}
 
function toggleInput(id){
  const el=document.getElementById(id+'-row');
  hideInputs();
  el.classList.toggle('show');
  const h=document.getElementById('hint');
  if(id==='li'){h.textContent='Tip: In OBS, use Recording Format → MKV or FLV (not MP4). MP4 only becomes readable after recording stops.';h.style.display='block'}
  else if(id==='su'){h.textContent='Tip: In OBS, set Stream → Custom → URL to rtmp://server:1935/live (requires RTMP server like mediamtx). Or use VLC → Stream output.';h.style.display='block'}
  else{h.style.display='none'}
}
 
function hideInputs(){
  document.getElementById('li-row').classList.remove('show');
  document.getElementById('su-row').classList.remove('show');
  document.getElementById('hint').style.display='none';
}
 
async function startLive(){
  const p=document.getElementById('li-path').value.trim();
  if(p)await api('source/livefile',{file_path:p});
  hideInputs();
  if(audioEnabled){setTimeout(restartAudio,1500)}
}
 
async function startStream(){
  const u=document.getElementById('su-url').value.trim();
  if(u)await api('source/streamurl',{url:u});
  hideInputs();
  if(audioEnabled){setTimeout(restartAudio,1500)}
}
 
// Audio functions
function initAudio(){
  audioPlayer=document.getElementById('audio-player');
  audioPlayer.volume=0.8;
}
 
function toggleAudio(){
  const btn=document.getElementById('audio-toggle');
  const ind=document.getElementById('audio-ind');
  
  if(audioEnabled){
    // Disable audio
    audioEnabled=false;
    if(audioPlayer){
      audioPlayer.pause();
      audioPlayer.src='';
    }
    btn.textContent='🔇';
    btn.classList.remove('active');
    btn.classList.add('muted');
    ind.classList.remove('active');
  }else{
    // Enable audio
    audioEnabled=true;
    startAudio();
    btn.textContent='🔊';
    btn.classList.add('active');
    btn.classList.remove('muted');
    ind.classList.add('active');
  }
}
 
function startAudio(){
  if(!audioPlayer)initAudio();
  audioPlayer.src='/audio?t='+Date.now();
  audioPlayer.play().catch(e=>console.log('Audio autoplay blocked:',e));
}
 
function restartAudio(){
  if(audioEnabled&&audioPlayer){
    audioPlayer.pause();
    audioPlayer.src='/audio?t='+Date.now();
    audioPlayer.play().catch(e=>console.log('Audio restart failed:',e));
  }
}
 
function setVolume(v){
  if(audioPlayer)audioPlayer.volume=v/100;
  document.getElementById('vol-pct').textContent=v+'%';
}
 
async function upd(){
  try{
    const d=await(await fetch('/api/status')).json();
    cSrc=d.source;
    document.getElementById('cur-vid').textContent=d.current_video;
    document.getElementById('src-lbl').textContent=d.source;
    ['playlist','screen','livefile','streamurl'].forEach(s=>{
      const e=document.getElementById('sb-'+s);
      if(e)e.className='src-btn'+(d.source===s?' active':'');
    });
    const isP=d.source==='playlist';
    document.getElementById('tp-sec').style.display=isP?'flex':'none';
    document.getElementById('prog-sec').style.display=isP?'block':'none';
    document.getElementById('pos-lbl').textContent=isP?'Video '+d.playlist_position+' of '+d.playlist_total:d.source==='screen'?'Capturing live screen':d.source==='streamurl'?'Receiving live stream':'Streaming live file';
    const b=document.getElementById('st-badge');
    b.textContent=d.state;
    b.className='badge state-'+d.state;
    document.getElementById('dot').style.background=d.state==='playing'?'#10b981':d.state==='paused'?'#f59e0b':'#ef4444';
    document.getElementById('pp-btn').textContent=d.state==='playing'?'⏸':'▶';
    if(isP){
      document.getElementById('pfill').style.width=d.progress_percent+'%';
      document.getElementById('t-cur').textContent=fmt(d.position_seconds);
      document.getElementById('t-tot').textContent=fmt(d.duration_seconds);
    }
    if(!d.mss_available){
      const e=document.getElementById('sb-screen');
      e.title='pip install mss required';
      e.style.opacity='.5';
    }
    // Update audio indicator
    const ind=document.getElementById('audio-ind');
    if(d.has_audio&&audioEnabled){
      ind.classList.add('active');
    }
  }catch(e){}
}
 
async function pl(){
  try{
    const d=await(await fetch('/api/playlist')).json();
    document.getElementById('pl').innerHTML=d.playlist.map((f,i)=>'<div class="pl-item '+(i===d.current_index&&cSrc==='playlist'?'active':'')+'" onclick="jumpTo('+i+')"><span>'+(i+1)+'. '+f+'</span>'+(i===d.current_index&&cSrc==='playlist'?'<span>▶</span>':'')+'</div>').join('');
  }catch(e){}
}
 
// Initialize
initAudio();
upd();
pl();
setInterval(upd,1000);
setInterval(pl,5000);
</script>
</body>
</html>
"""

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Corporate TV — Streaming Server (v5 with Audio)")
    print(f"  Dashboard:    http://localhost:{SERVER_PORT}")
    print(f"  Video Stream: http://localhost:{SERVER_PORT}/stream")
    print(f"  Audio Stream: http://localhost:{SERVER_PORT}/audio")
    print(f"  Media dir:    {os.path.abspath(MEDIA_DIR)}")
    print(
        f"  Screen (mss): {'Available' if MSS_AVAILABLE else 'Not installed (pip install mss)'}"
    )
    print("=" * 60)
    print()
    print("  Sources:")
    print("    POST /api/source/playlist")
    print("    POST /api/source/screen")
    print('    POST /api/source/livefile   {"file_path": "/path/to/recording.mkv"}')
    print('    POST /api/source/streamurl  {"url": "rtmp://localhost:1935/live"}')
    print()
    print("  OBS Tips:")
    print("    - For livefile: Set Recording Format to MKV or FLV (not MP4)")
    print("    - For streamurl: Stream to rtmp://server:1935/live (needs RTMP server)")
    print()
    print("  Audio:")
    print("    - Click the speaker icon in the dashboard to enable audio")
    print("    - Audio is extracted from video sources using FFmpeg")
    print("    - Screen capture does not include audio by default")
    print()
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
