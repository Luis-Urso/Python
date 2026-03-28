#!/usr/bin/env python3
"""
Corporate TV Player for MediaMTX (OpenCV Version)
==================================================
A reliable, full-screen video player for corporate displays.
Connects to MediaMTX server and plays streams with audio using OpenCV.

Requirements:
    pip install opencv-python numpy pyaudio requests screeninfo

Usage:
    python IPTV_Player_V2.py --server localhost --stream test
    python IPTV_Player_V2.py --server 192.168.1.100 --stream corporate-tv --fullscreen
    python IPTV_Player_V2.py --resolution 1920x1080 --server localhost --stream test
"""

import sys
import argparse
import time
import threading
import logging
import queue
import subprocess
import signal
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from screeninfo import get_monitors

    HAS_SCREENINFO = True
except ImportError:
    HAS_SCREENINFO = False
    print("WARNING: screeninfo not installed. Auto screen detection disabled.")
    print("         Install with: pip install screeninfo")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PlayerConfig:
    """Configuration for the Corporate TV Player."""

    server: str = "localhost"
    stream: str = "test"
    protocol: str = "rtmp"
    resolution: Optional[Tuple[int, int]] = None
    fullscreen: bool = False
    volume: int = 100
    reconnect_delay: int = 3
    max_reconnect_attempts: int = 50
    buffer_size: int = 5
    show_stats: bool = False


class AudioPlayer:
    """
    Handles audio playback using FFplay subprocess.
    Runs FFplay in a separate process to decode and play audio.
    FFplay is more reliable for audio playback than FFmpeg piping.
    """

    def __init__(self, url: str, volume: int = 100):
        self.url = url
        self.volume = volume
        self.process: Optional[subprocess.Popen] = None
        self.running = False

    def start(self):
        """Start audio playback."""
        if self.process:
            self.stop()

        # Calculate FFplay volume (0-100)
        vol = self.volume

        # Use FFplay for audio - much more reliable across platforms
        # FFplay handles audio output automatically on all platforms
        cmd = [
            "ffplay",
            "-nodisp",  # No video display (audio only)
            "-autoexit",  # Exit when stream ends
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",  # Drop frames if needed
            "-vn",  # No video processing
            "-volume",
            str(vol),  # Volume 0-100
            "-i",
            self.url,
        ]

        # Alternative: Use FFmpeg with DirectShow on Windows if FFplay doesn't work
        cmd_ffmpeg_windows = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self.url,
            "-vn",
            "-af",
            f"volume={vol / 100.0}",
            "-f",
            "sdl2",  # SDL2 audio output (works on Windows)
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "audio",  # SDL window title
        ]

        try:
            # First try FFplay (preferred, works on all platforms)
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0,
            )
            self.running = True
            logger.info("Audio playback started (using ffplay)")

            # Check if process started successfully
            time.sleep(0.5)
            if self.process.poll() is not None:
                # FFplay failed, try FFmpeg with SDL2 on Windows
                if sys.platform == "win32":
                    logger.info("FFplay failed, trying FFmpeg with SDL2...")
                    self.process = subprocess.Popen(
                        cmd_ffmpeg_windows,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    self.running = True
                    logger.info("Audio playback started (using ffmpeg sdl2)")

        except FileNotFoundError:
            logger.warning("FFplay/FFmpeg not found. Audio playback disabled.")
            logger.warning("Install FFmpeg: https://ffmpeg.org/download.html")
            logger.warning("Make sure ffplay.exe is in your PATH")
        except Exception as e:
            logger.error(f"Failed to start audio: {e}")

    def stop(self):
        """Stop audio playback."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping audio: {e}")
            finally:
                self.process = None
        logger.info("Audio playback stopped")

    def set_volume(self, volume: int):
        """Set volume (requires restart)."""
        self.volume = max(0, min(100, volume))
        if self.running:
            url = self.url
            self.stop()
            self.url = url
            self.start()

    def is_running(self) -> bool:
        """Check if audio is still playing."""
        if self.process:
            return self.process.poll() is None
        return False


class StreamReader:
    """
    Reads video frames from MediaMTX stream using OpenCV.
    Runs in a separate thread for smooth playback.
    """

    def __init__(self, url: str, buffer_size: int = 5):
        self.url = url
        self.buffer_size = buffer_size
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.connected = False
        self.error_message = ""

    def start(self):
        """Start reading frames from the stream."""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop reading frames."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()
            self.capture = None
        self.connected = False
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _read_loop(self):
        """Main loop for reading frames (runs in thread)."""
        while self.running:
            try:
                if not self.capture or not self.capture.isOpened():
                    self._connect()
                    continue

                ret, frame = self.capture.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame, reconnecting...")
                    self.connected = False
                    self._connect()
                    continue

                self.connected = True
                self.error_message = ""

                # Add frame to queue, drop oldest if full
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()  # Drop oldest
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass

            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                self.error_message = str(e)
                self.connected = False
                time.sleep(1)

    def _connect(self):
        """Connect to the stream."""
        logger.info(f"Connecting to stream: {self.url}")

        if self.capture:
            self.capture.release()

        # OpenCV VideoCapture with optimized settings
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        # Set buffer size to reduce latency
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to reduce latency with FFmpeg options
        # These are set via environment variable before capture

        if self.capture.isOpened():
            self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                f"Connected! Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}"
            )
            self.connected = True
        else:
            logger.warning("Failed to connect to stream")
            self.error_message = "Failed to connect"
            self.connected = False
            time.sleep(2)  # Wait before retry

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None


class CorporateTVPlayer:
    """
    Full-screen video player for corporate TV displays.
    Uses OpenCV for video and FFmpeg for audio.
    """

    def __init__(self, config: PlayerConfig):
        self.config = config
        self.stream_reader: Optional[StreamReader] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.running = False
        self.window_name = "Corporate TV"
        self.reconnect_attempts = 0
        self.last_frame: Optional[np.ndarray] = None
        self.display_width = 1920
        self.display_height = 1080

        # Get display resolution
        self._detect_resolution()

        # Build stream URL
        self.stream_url = self._build_stream_url()

        logger.info(f"Display resolution: {self.display_width}x{self.display_height}")
        logger.info(f"Stream URL: {self.stream_url}")

    def _detect_resolution(self):
        """Detect or set display resolution."""
        if self.config.resolution:
            self.display_width, self.display_height = self.config.resolution
            logger.info(
                f"Using manual resolution: {self.display_width}x{self.display_height}"
            )
        elif HAS_SCREENINFO:
            try:
                monitors = get_monitors()
                if monitors:
                    primary = monitors[0]
                    self.display_width = primary.width
                    self.display_height = primary.height
                    logger.info(
                        f"Auto-detected resolution: {self.display_width}x{self.display_height}"
                    )
            except Exception as e:
                logger.warning(f"Could not detect screen resolution: {e}")

    def _build_stream_url(self) -> str:
        """Build the stream URL based on protocol."""
        server = self.config.server
        stream = self.config.stream
        protocol = self.config.protocol

        urls = {
            "rtmp": f"rtmp://{server}:1935/{stream}",
            "rtsp": f"rtsp://{server}:8554/{stream}",
            "hls": f"http://{server}:8888/{stream}/index.m3u8",
            "srt": f"srt://{server}:8890?streamid=read:{stream}",
        }

        return urls.get(protocol, urls["rtmp"])

    def _create_status_frame(
        self, message: str, color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Create a frame with status message."""
        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Add corporate styling
        # Dark gradient background
        for y in range(self.display_height):
            intensity = int(20 + (y / self.display_height) * 15)
            frame[y, :] = (intensity, intensity, intensity)

        # Status message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (self.display_width - text_size[0]) // 2
        text_y = (self.display_height + text_size[1]) // 2

        cv2.putText(
            frame,
            message,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Stream info at bottom
        info_text = f"Stream: {self.config.stream} | Server: {self.config.server} | Protocol: {self.config.protocol.upper()}"
        info_size = cv2.getTextSize(info_text, font, 0.6, 1)[0]
        info_x = (self.display_width - info_size[0]) // 2
        cv2.putText(
            frame,
            info_text,
            (info_x, self.display_height - 30),
            font,
            0.6,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _create_overlay(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """Add status overlay to frame."""
        if not self.config.show_stats:
            return frame

        overlay = frame.copy()

        # Semi-transparent background for stats
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Status indicator
        status_color = (0, 255, 0) if stats.get("connected") else (0, 165, 255)
        status_text = "● LIVE" if stats.get("connected") else "● CONNECTING"
        cv2.putText(
            frame, status_text, (20, 40), font, 0.7, status_color, 2, cv2.LINE_AA
        )

        # FPS
        fps_text = f"FPS: {stats.get('fps', 0):.1f}"
        cv2.putText(
            frame, fps_text, (20, 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Resolution
        res_text = f"Resolution: {stats.get('width', 0)}x{stats.get('height', 0)}"
        cv2.putText(
            frame, res_text, (20, 90), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Audio status
        audio_color = (0, 255, 0) if stats.get("audio") else (0, 0, 255)
        audio_text = "Audio: ON" if stats.get("audio") else "Audio: OFF"
        cv2.putText(
            frame, audio_text, (20, 110), font, 0.5, audio_color, 1, cv2.LINE_AA
        )

        return frame

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display while maintaining aspect ratio."""
        h, w = frame.shape[:2]

        # Calculate scaling factor
        scale_w = self.display_width / w
        scale_h = self.display_height / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create black canvas and center the frame
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        x_offset = (self.display_width - new_w) // 2
        y_offset = (self.display_height - new_h) // 2

        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def start(self):
        """Start the player."""
        self.running = True
        self.was_connected = False  # Track previous connection state
        self.audio_restart_pending = False

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if self.config.fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        # Start stream reader
        self.stream_reader = StreamReader(self.stream_url, self.config.buffer_size)
        self.stream_reader.start()

        # Start audio player
        self.audio_player = AudioPlayer(self.stream_url, self.config.volume)
        self.audio_player.start()

        # Main display loop
        self._display_loop()

    def _check_audio_health(self):
        """Monitor audio player and restart if needed."""
        if not self.audio_player:
            return

        # Check if audio process has died
        if not self.audio_player.is_running():
            # Only restart if video is connected
            if self.stream_reader and self.stream_reader.connected:
                logger.info("Audio stopped, restarting...")
                time.sleep(0.5)  # Brief delay before restart
                self.audio_player.stop()
                self.audio_player = AudioPlayer(self.stream_url, self.config.volume)
                self.audio_player.start()

    def _display_loop(self):
        """Main loop for displaying frames."""
        frame_time = 1.0 / 30  # Target 30 FPS for display
        last_frame_time = time.time()
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0.0
        last_audio_check = time.time()
        audio_check_interval = 2.0  # Check audio every 2 seconds

        while self.running:
            current_time = time.time()

            # Get frame from reader
            frame = None
            if self.stream_reader:
                frame = self.stream_reader.get_frame()

            # Use last frame if no new frame available
            if frame is not None:
                self.last_frame = frame
                fps_counter += 1

            # Calculate FPS every second
            if current_time - fps_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = current_time

            # Check for stream reconnection and restart audio
            is_connected = self.stream_reader.connected if self.stream_reader else False

            # Detect reconnection: was disconnected, now connected
            if is_connected and not self.was_connected:
                logger.info("Stream reconnected, restarting audio...")
                time.sleep(1.0)  # Wait for stream to stabilize
                if self.audio_player:
                    self.audio_player.stop()
                self.audio_player = AudioPlayer(self.stream_url, self.config.volume)
                self.audio_player.start()

            # Periodic audio health check
            if current_time - last_audio_check >= audio_check_interval:
                self._check_audio_health()
                last_audio_check = current_time

            self.was_connected = is_connected

            # Prepare display frame
            if self.last_frame is not None:
                display_frame = self._resize_frame(self.last_frame)

                # Add overlay if enabled
                stats = {
                    "connected": self.stream_reader.connected
                    if self.stream_reader
                    else False,
                    "fps": current_fps,
                    "width": self.stream_reader.frame_width
                    if self.stream_reader
                    else 0,
                    "height": self.stream_reader.frame_height
                    if self.stream_reader
                    else 0,
                    "audio": self.audio_player.is_running()
                    if self.audio_player
                    else False,
                }
                display_frame = self._create_overlay(display_frame, stats)
            else:
                # Show connecting message
                if self.stream_reader and self.stream_reader.error_message:
                    msg = f"Connecting... ({self.stream_reader.error_message})"
                else:
                    msg = "Connecting to stream..."
                display_frame = self._create_status_frame(msg, (0, 165, 255))

            # Display frame
            cv2.imshow(self.window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

            # Check if window was closed
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break

            # Frame rate control
            elapsed = time.time() - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()

        self._cleanup()

    def _handle_key(self, key: int):
        """Handle keyboard input."""
        if key == ord("q") or key == 27:  # Q or ESC
            self.running = False

        elif key == ord("f"):  # Toggle fullscreen
            current = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
            if current == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                )
                cv2.resizeWindow(
                    self.window_name, self.display_width, self.display_height
                )
            else:
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )

        elif key == ord("r"):  # Reconnect
            logger.info("Manual reconnect requested")
            self._reconnect()

        elif key == ord("s"):  # Toggle stats
            self.config.show_stats = not self.config.show_stats
            logger.info(f"Stats display: {'ON' if self.config.show_stats else 'OFF'}")

        elif key == ord("+") or key == ord("="):  # Volume up
            if self.audio_player:
                new_vol = min(100, self.config.volume + 10)
                self.config.volume = new_vol
                logger.info(f"Volume: {new_vol}%")
                # Note: Requires audio restart to take effect

        elif key == ord("-"):  # Volume down
            if self.audio_player:
                new_vol = max(0, self.config.volume - 10)
                self.config.volume = new_vol
                logger.info(f"Volume: {new_vol}%")

        elif key == ord("m"):  # Mute toggle
            if self.audio_player:
                if self.audio_player.is_running():
                    self.audio_player.stop()
                    logger.info("Audio muted")
                else:
                    self.audio_player.start()
                    logger.info("Audio unmuted")

    def _reconnect(self):
        """Reconnect to the stream."""
        logger.info("Reconnecting...")

        # Stop current readers
        if self.stream_reader:
            self.stream_reader.stop()
        if self.audio_player:
            self.audio_player.stop()

        self.last_frame = None
        time.sleep(1)

        # Restart
        self.stream_reader = StreamReader(self.stream_url, self.config.buffer_size)
        self.stream_reader.start()

        self.audio_player = AudioPlayer(self.stream_url, self.config.volume)
        self.audio_player.start()

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Shutting down player...")

        if self.stream_reader:
            self.stream_reader.stop()

        if self.audio_player:
            self.audio_player.stop()

        cv2.destroyAllWindows()

    def stop(self):
        """Stop the player."""
        self.running = False


def parse_resolution(res_string: str) -> Tuple[int, int]:
    """Parse resolution string like '1920x1080' into tuple."""
    try:
        width, height = res_string.lower().split("x")
        return (int(width), int(height))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format: {res_string}. Use format: 1920x1080"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Corporate TV Player for MediaMTX streams (OpenCV version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --server localhost --stream test
  %(prog)s --server 192.168.1.100 --stream corporate-tv --fullscreen
  %(prog)s --server 10.0.0.50 --stream lobby-display --resolution 1920x1080 --protocol rtsp

Keyboard Controls:
  F          Toggle fullscreen
  R          Reconnect to stream
  S          Toggle stats overlay
  +/-        Volume up/down
  M          Mute/unmute audio
  Q/Escape   Quit

Supported Protocols:
  rtmp       RTMP (default, lowest latency)
  rtsp       RTSP (good compatibility)
  hls        HLS (highest compatibility, higher latency)
  srt        SRT (good for unreliable networks)

Requirements:
  - Python packages: opencv-python, numpy, requests, screeninfo
  - FFmpeg (for audio playback): brew install ffmpeg (macOS) or apt install ffmpeg (Linux)
        """,
    )

    parser.add_argument(
        "--server",
        "-s",
        default="localhost",
        help="MediaMTX server address (default: localhost)",
    )

    parser.add_argument(
        "--stream", "-n", default="test", help="Stream name/path (default: test)"
    )

    parser.add_argument(
        "--protocol",
        "-p",
        choices=["rtmp", "rtsp", "hls", "srt"],
        default="rtmp",
        help="Streaming protocol (default: rtmp)",
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=parse_resolution,
        help="Window resolution, e.g., 1920x1080 (default: auto-detect)",
    )

    parser.add_argument(
        "--fullscreen", "-f", action="store_true", help="Start in fullscreen mode"
    )

    parser.add_argument(
        "--volume",
        "-v",
        type=int,
        default=100,
        choices=range(0, 101),
        metavar="0-100",
        help="Initial volume level (default: 100)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show stats overlay (FPS, resolution, connection status)",
    )

    parser.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Frame buffer size (default: 5, lower = less latency)",
    )

    args = parser.parse_args()

    # Build configuration
    config = PlayerConfig(
        server=args.server,
        stream=args.stream,
        protocol=args.protocol,
        resolution=args.resolution,
        fullscreen=args.fullscreen,
        volume=args.volume,
        show_stats=args.stats,
        buffer_size=args.buffer,
    )

    logger.info("=" * 50)
    logger.info("Corporate TV Player (OpenCV) Starting")
    logger.info("=" * 50)
    logger.info(f"Server: {config.server}")
    logger.info(f"Stream: {config.stream}")
    logger.info(f"Protocol: {config.protocol}")
    logger.info(f"Fullscreen: {config.fullscreen}")
    logger.info("=" * 50)
    logger.info("Controls: F=fullscreen, R=reconnect, S=stats, Q=quit")
    logger.info("=" * 50)

    # Create and start player
    player = CorporateTVPlayer(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        player.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start player
    player.start()


if __name__ == "__main__":
    main()
