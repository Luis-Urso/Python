# IPTV Server and Client Configuration Guide

## Corporate TV Streaming System
### MediaMTX + OBS Studio + Python Client

**Version 1.2**  
**March 2026**

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [MediaMTX Server Configuration](#2-mediamtx-server-configuration)
3. [OBS Studio Configuration](#3-obs-studio-configuration)
4. [Corporate TV Player (Client Application)](#4-corporate-tv-player-client-application)
5. [Browser-Based Playback (WebRTC and HLS)](#5-browser-based-playback-webrtc-and-hls)
6. [Troubleshooting Guide](#6-troubleshooting-guide)
7. [Quick Reference](#7-quick-reference)
8. [Document Version History](#8-document-version-history)

---

## 1. System Architecture

The Corporate TV streaming system uses a multi-component architecture to deliver live video content from OBS Studio to various display devices. MediaMTX acts as the central hub, receiving streams and distributing them via multiple protocols.

### 1.1 Architecture Diagram

```
┌─────────────────┐          ┌─────────────────────────────────────┐
│                 │   RTMP   │                                     │
│   OBS Studio    │─────────▶│         MediaMTX Server             │
│  (Video Source) │  :1935   │    (Multi-Protocol Streaming)       │
│                 │          │                                     │
└─────────────────┘          └──────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │  RTMP/RTSP        │   │  WebRTC           │   │  HLS              │
        │  :1935 / :8554    │   │  :8889            │   │  :8888            │
        └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │                       │
                  ▼                       ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │ Corporate TV      │   │ Web Browser       │   │ Web Browser       │
        │ Player (Python)   │   │ (Low Latency)     │   │ (High Compat.)    │
        │ NUC / Mac Mini    │   │ Desktop/Mobile    │   │ Any Device        │
        └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │                       │
                  ▼                       ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   TV Displays     │   │     Desktop       │   │  Mobile/Tablet    │
        └───────────────────┘   └───────────────────┘   └───────────────────┘
```

### 1.2 Component Overview

| Component | Role | Location |
|-----------|------|----------|
| OBS Studio | Captures screen/camera content and pushes RTMP to MediaMTX | Streaming PC |
| MediaMTX | Multi-protocol streaming server (RTMP, RTSP, HLS, WebRTC) | Server |
| Corporate TV Player | Python/OpenCV client for fullscreen display on TVs | NUC / Mac Mini |
| Web Browser | WebRTC or HLS playback for desktop and mobile devices | Any device |

### 1.3 Data Flow

```
OBS Studio --RTMP--> MediaMTX (:1935) --RTMP/RTSP/HLS/WebRTC--> Clients
```

### 1.4 Supported Protocols

| Protocol | Port | Latency | Best For |
|----------|------|---------|----------|
| RTMP | 1935 | 1-2 seconds | Primary OBS input, Python player |
| RTSP | 8554 | 1-2 seconds | LAN environments, VLC playback |
| HLS | 8888 | 3-5 seconds | Browser playback, high compatibility |
| WebRTC | 8889 | ~0.5 seconds | Ultra-low latency browser playback |
| SRT | 8890 | 1-2 seconds | Unreliable networks, packet loss tolerance |
| API | 9997 | N/A | Monitoring and management |

---

## 2. MediaMTX Server Configuration

MediaMTX is a ready-to-use, zero-dependency media server that supports multiple streaming protocols. It acts as the central hub between your video source (OBS) and display devices.

### 2.1 Installation

#### macOS

**Option A: Using Homebrew (Recommended)**

```bash
brew install mediamtx
```

**Option B: Manual Download**

- Download from: https://github.com/bluenviron/mediamtx/releases
- Choose `mediamtx_vX.X.X_darwin_arm64.tar.gz` (Apple Silicon) or `darwin_amd64.tar.gz` (Intel)

```bash
tar -xf mediamtx_darwin_arm64.tar.gz
chmod +x mediamtx
./mediamtx
```

#### Linux (Ubuntu/Debian)

```bash
# Download latest release
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.17.0_linux_amd64.tar.gz

# Extract
tar -xf mediamtx_v1.17.0_linux_amd64.tar.gz

# Move to system location
sudo mv mediamtx /usr/local/bin/
sudo mv mediamtx.yml /usr/local/etc/

# Run
mediamtx
```

#### Windows

- Download `mediamtx_vX.X.X_windows_amd64.zip` from GitHub releases
- Extract the ZIP file to a folder (e.g., `C:\MediaMTX`)
- Open Command Prompt or PowerShell in that folder
- Run: `mediamtx.exe`

> **Note:** Windows Firewall may prompt you to allow network access. Click Allow for both Private and Public networks.

#### Docker (any OS)

```bash
docker run --rm -it --network=host bluenviron/mediamtx:latest
```

### 2.2 Configuration File

MediaMTX uses a YAML configuration file (mediamtx.yml). Default locations:

| Platform | Default Location |
|----------|------------------|
| macOS/Linux | ./mediamtx.yml (same folder as binary) |
| Homebrew | /opt/homebrew/etc/mediamtx.yml |
| Windows | mediamtx.yml (same folder as .exe) |

#### Recommended Configuration

```yaml
###############################################
# MediaMTX Configuration for Corporate TV
###############################################

# API Configuration
api: yes
apiAddress: :9997

# Logging
logLevel: info
logDestinations: [stdout]

# RTSP Configuration
rtspAddress: :8554
protocols: [tcp, udp]

# RTMP Configuration
rtmpAddress: :1935

# HLS Configuration
hlsAddress: :8888
hlsAlwaysRemux: yes
hlsSegmentCount: 3
hlsSegmentDuration: 1s

# WebRTC Configuration
webrtcAddress: :8889

# SRT Configuration
srtAddress: :8890

# Path Configuration
paths:
  all_others:
```

> **Tip:** The 'all_others' path accepts any stream name. You can define specific paths like 'lobby-tv' or 'conference-room' for better control.

### 2.3 Starting MediaMTX

#### Manual Start

```bash
# Navigate to MediaMTX folder and run:
./mediamtx

# Or with custom config:
./mediamtx /path/to/mediamtx.yml
```

When MediaMTX starts successfully, you should see:

```
2026/03/26 10:00:00 INF MediaMTX v1.17.0
2026/03/26 10:00:00 INF [RTSP] listener opened on :8554
2026/03/26 10:00:00 INF [RTMP] listener opened on :1935
2026/03/26 10:00:00 INF [HLS] listener opened on :8888
2026/03/26 10:00:00 INF [WebRTC] listener opened on :8889
2026/03/26 10:00:00 INF [SRT] listener opened on :8890
2026/03/26 10:00:00 INF [API] listener opened on :9997
```

#### Run as Background Service (Linux systemd)

Create `/etc/systemd/system/mediamtx.service`:

```ini
[Unit]
Description=MediaMTX Streaming Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/mediamtx /usr/local/etc/mediamtx.yml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mediamtx
sudo systemctl start mediamtx
```

#### Run as Background Service (macOS launchd)

Create `~/Library/LaunchAgents/com.mediamtx.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.mediamtx</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/mediamtx</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.mediamtx.plist
```

#### Run as Windows Service (with Power Recovery)

For Windows servers that need MediaMTX to automatically restart after a power outage, follow these steps:

**Step 1: Configure BIOS for Power Recovery**

- Enter BIOS/UEFI during boot
- Find 'AC Power Recovery' or 'After Power Loss' setting
- Set to 'Power On' to auto-start when power returns
- Save and exit BIOS

**Step 2: Install NSSM (Non-Sucking Service Manager)**

NSSM allows running any executable as a Windows service:

- Download NSSM from https://nssm.cc/download
- Extract to `C:\nssm`
- Add `C:\nssm\win64` to your system PATH

**Step 3: Create MediaMTX Windows Service**

Open Command Prompt as Administrator and run:

```batch
nssm install MediaMTX C:\MediaMTX\mediamtx.exe
nssm set MediaMTX AppDirectory C:\MediaMTX
nssm set MediaMTX DisplayName "MediaMTX Streaming Server"
nssm set MediaMTX Description "Multi-protocol streaming server for Corporate TV"
nssm set MediaMTX Start SERVICE_AUTO_START
nssm set MediaMTX AppStdout C:\MediaMTX\logs\service.log
nssm set MediaMTX AppStderr C:\MediaMTX\logs\error.log
nssm set MediaMTX AppRotateFiles 1
nssm set MediaMTX AppRotateBytes 1048576
```

Create the logs directory:

```batch
mkdir C:\MediaMTX\logs
```

**Step 4: Configure Service Recovery Options**

```batch
nssm set MediaMTX AppExit Default Restart
nssm set MediaMTX AppRestartDelay 5000
```

**Step 5: Start the Service**

```batch
nssm start MediaMTX
```

**Step 6: Verify Service Status**

```batch
nssm status MediaMTX

REM Or use Windows Services
services.msc
```

**Managing the Service**

```batch
REM Stop the service
nssm stop MediaMTX

REM Restart the service
nssm restart MediaMTX

REM Remove the service (if needed)
nssm remove MediaMTX confirm
```

> **Tip:** The service will automatically start when Windows boots and restart if it crashes. Combined with BIOS power recovery, MediaMTX will be available after any power outage.

#### Alternative: Windows Task Scheduler Method

If you prefer not to use NSSM, you can use Task Scheduler:

- Open Task Scheduler (taskschd.msc)
- Create Task (not Basic Task)
- General: Name 'MediaMTX', select 'Run whether user is logged on or not'
- General: Check 'Run with highest privileges'
- Triggers: Add 'At startup'
- Actions: Start a program > `C:\MediaMTX\mediamtx.exe`
- Settings: Check 'If the task fails, restart every 1 minute'
- Settings: Set 'Attempt to restart up to 999 times'

> **Note:** The NSSM method is recommended as it provides better process management and logging capabilities.

### 2.4 Testing MediaMTX

**Test 1: Verify Server is Running**

Open your browser and navigate to:

```
http://localhost:9997/v3/paths/list
```

Expected response (empty paths list):

```json
{
  "itemCount": 0,
  "pageCount": 0,
  "items": []
}
```

**Test 2: Publish a Test Stream with FFmpeg**

```bash
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
  -f lavfi -i sine=frequency=1000:sample_rate=44100 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -c:a aac -b:a 128k \
  -f flv rtmp://localhost:1935/test
```

**Test 3: Play Stream in Browser**

- WebRTC (low latency): http://localhost:8889/test
- HLS (high compatibility): http://localhost:8888/test

---

## 3. OBS Studio Configuration

OBS Studio captures the content you want to broadcast (screen, camera, presentations) and pushes it as an RTMP stream to MediaMTX.

### 3.1 Stream Settings

Go to Settings > Stream and configure:

| Setting | Value |
|---------|-------|
| Service | Custom... |
| Server | rtmp://SERVER_IP:1935 |
| Stream Key | your-stream-name (e.g., corporate-tv, lobby, test) |

> **Important:** Do NOT include /live or any path in the Server URL. The Stream Key alone determines the path. Use simple, lowercase names without spaces.

### 3.2 Output Settings

Go to Settings > Output, set Output Mode to Advanced, then configure the Streaming tab:

| Setting | Value | Notes |
|---------|-------|-------|
| Encoder | x264 | Software encoder - most compatible |
| Rate Control | CBR | Constant Bit Rate - predictable bandwidth |
| Bitrate | 2500 Kbps | Good balance; increase to 4000-6000 for higher quality |
| Keyframe Interval | 2 seconds | CRITICAL for low latency! Must be 1-2 seconds |
| CPU Usage Preset | veryfast | Lower CPU usage; use 'faster' if CPU allows |
| Profile | baseline | Maximum compatibility with all decoders |
| Tune | zerolatency | Optimizes encoding for minimum delay |
| Audio Encoder | AAC | Standard audio codec |
| Audio Bitrate | 160 Kbps | Good quality without excessive bandwidth |

> **Important:** The Keyframe Interval setting is critical! A value higher than 2 seconds will cause significant playback delays and buffering issues.

### 3.3 Video Settings

Go to Settings > Video:

| Setting | Value |
|---------|-------|
| Base (Canvas) Resolution | 1920x1080 |
| Output (Scaled) Resolution | 1920x1080 or 1280x720 |
| Downscale Filter | Lanczos (Sharpened scaling, 36 samples) |
| Common FPS Values | 30 (use 60 only for fast-motion content) |

#### Recommended Resolution and Bitrate Combinations

| Resolution | FPS | Bitrate | Use Case |
|------------|-----|---------|----------|
| 1920x1080 | 30 | 4000-6000 Kbps | High quality |
| 1920x1080 | 30 | 2500-3500 Kbps | Recommended (balanced) |
| 1280x720 | 30 | 1500-2500 Kbps | Limited bandwidth |

### 3.4 Audio Settings

Go to Settings > Audio:

| Setting | Value |
|---------|-------|
| Sample Rate | 44.1 kHz |
| Channels | Stereo (use Mono for single microphone) |

> **Tip:** If you don't need audio, you can mute all sources in the Audio Mixer. The stream will still work without audio.

### 3.5 WHIP Streaming (OBS 30+)

For ultra-low latency WebRTC streaming with native Opus audio support, OBS Studio 30.0+ supports WHIP (WebRTC-HTTP Ingestion Protocol). This bypasses the AAC/Opus codec compatibility issue completely.

**To use WHIP:**

- Go to Settings > Stream
- Set Service to WHIP
- Set Server to: `http://localhost:8889/your-stream/whip`
- Leave Bearer Token empty unless authentication is enabled

> **Tip:** WHIP provides the lowest latency (~0.5 seconds) and native WebRTC audio support. Recommended if you have OBS 30.0 or higher.

### 3.6 Pre-Flight Checklist

Before clicking Start Streaming, verify:

- [ ] MediaMTX server is running
- [ ] Server URL is correct (rtmp://SERVER:1935)
- [ ] Stream Key matches your player configuration
- [ ] Keyframe Interval is set to 2 seconds
- [ ] Video sources are added and visible in preview
- [ ] Audio levels show activity (if audio is needed)

### 3.7 OBS Auto-Start with Streaming (Power Recovery)

For unattended streaming setups, OBS can be configured to automatically start and begin streaming when the computer boots. This is essential for scenarios where the streaming PC may lose power and needs to resume automatically.

#### Windows Setup

**Step 1: Configure BIOS for Power Recovery**

- Enter BIOS/UEFI during boot
- Set 'AC Power Recovery' to 'Power On'
- Save and exit

**Step 2: Configure Windows Auto-Login**

- Press Win + R, type 'netplwiz', press Enter
- Uncheck 'Users must enter a user name and password'
- Click Apply and enter credentials

**Step 3: Create OBS Startup Script**

Create a batch file `C:\OBS\start_obs_streaming.bat`:

```batch
@echo off
REM OBS Auto-Start Streaming Script
REM Wait for system and network to initialize
timeout /t 30 /nobreak

REM Wait for MediaMTX to be available (optional check)
:waitserver
ping -n 1 YOUR_MEDIAMTX_SERVER_IP >nul 2>&1
if errorlevel 1 (
    echo Waiting for MediaMTX server...
    timeout /t 5 /nobreak
    goto waitserver
)

REM Start OBS with streaming enabled
cd /d "C:\Program Files\obs-studio\bin\64bit"
start "" obs64.exe --startstreaming --minimize-to-tray

REM Keep script running to restart OBS if it crashes
:monitor
timeout /t 60 /nobreak
tasklist /FI "IMAGENAME eq obs64.exe" 2>NUL | find /I "obs64.exe" >NUL
if errorlevel 1 (
    echo OBS crashed, restarting...
    start "" obs64.exe --startstreaming --minimize-to-tray
)
goto monitor
```

**Step 4: OBS Command Line Parameters**

| Parameter | Description |
|-----------|-------------|
| --startstreaming | Automatically start streaming when OBS launches |
| --startrecording | Automatically start recording when OBS launches |
| --minimize-to-tray | Start minimized to system tray |
| --scene "Scene Name" | Start with a specific scene |
| --profile "Profile Name" | Start with a specific profile |
| --collection "Collection" | Start with a specific scene collection |

**Step 5: Add to Windows Startup**

*Option A: Startup Folder*

- Press Win + R, type 'shell:startup', press Enter
- Create a shortcut to `start_obs_streaming.bat`
- Right-click shortcut > Properties > Run: Minimized

*Option B: Task Scheduler (Recommended)*

- Open Task Scheduler (taskschd.msc)
- Create Task > Name: 'OBS Auto-Stream'
- General: Run whether user is logged on or not
- Triggers: At startup, delay for 30 seconds
- Actions: Start program > `C:\OBS\start_obs_streaming.bat`
- Settings: If fails, restart every 1 minute

#### macOS Setup

Create a launch agent at `~/Library/LaunchAgents/com.obs.autostream.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.obs.autostream</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Applications/OBS.app/Contents/MacOS/OBS</string>
    <string>--startstreaming</string>
    <string>--minimize-to-tray</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key>
  <integer>30</integer>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
</dict>
</plist>
```

Load the launch agent:

```bash
launchctl load ~/Library/LaunchAgents/com.obs.autostream.plist
```

#### Linux Setup

Create a systemd user service at `~/.config/systemd/user/obs-autostream.service`:

```ini
[Unit]
Description=OBS Studio Auto-Streaming
After=network-online.target graphical-session.target
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/bin/sleep 30
ExecStart=/usr/bin/obs --startstreaming --minimize-to-tray
Restart=always
RestartSec=10
Environment=DISPLAY=:0

[Install]
WantedBy=graphical-session.target
```

Enable and start the service:

```bash
systemctl --user daemon-reload
systemctl --user enable obs-autostream
systemctl --user start obs-autostream
```

#### Important Considerations for OBS Auto-Start

- Ensure OBS profile and scene collection are properly configured before enabling auto-start
- Test the stream manually first to verify all sources work correctly
- Configure OBS to reconnect automatically: Settings > Advanced > Automatically reconnect
- Set a reasonable reconnect delay (e.g., 10 seconds) and max retries
- Consider using a static video source (image/video file) as fallback if capture fails
- Disable any OBS update prompts that might interrupt streaming

> **Important:** For production environments, test the complete power cycle: shut down, unplug power, restore power, and verify OBS starts streaming automatically.

---

## 4. Corporate TV Player (Client Application)

The Corporate TV Player is a Python/OpenCV-based client designed for fullscreen video display on dedicated hardware such as NUC computers or Mac Minis connected to TVs.

### 4.1 System Requirements

#### Hardware Requirements

- Intel NUC, Mac Mini, or similar compact PC
- Minimum 4GB RAM (8GB recommended)
- HDMI or DisplayPort output
- Stable network connection (wired Ethernet recommended)

#### Software Requirements

| Component | Requirement |
|-----------|-------------|
| Python | Version 3.8 or higher |
| FFmpeg | Required for audio playback |
| OpenCV | opencv-python >= 4.8.0 |
| Additional Libraries | numpy, requests, screeninfo |

### 4.2 Installation

#### Step 1: Install FFmpeg

**macOS:**

```bash
brew install ffmpeg
```

**Ubuntu/Debian Linux:**

```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**

Download FFmpeg from https://ffmpeg.org/download.html and add it to your system PATH.

#### Step 2: Install Python Dependencies

```bash
pip install opencv-python numpy requests screeninfo
```

### 4.3 Basic Usage

#### Quick Start

```bash
python IPTV_Player_v2.py --server localhost --stream test
```

#### Connect to Remote Server

```bash
python IPTV_Player_v2.py --server 192.168.1.100 --stream corporate-tv
```

#### Fullscreen Mode (Production)

```bash
python IPTV_Player_v2.py --server localhost --stream test --fullscreen
```

#### Custom Resolution

```bash
python IPTV_Player_v2.py --server localhost --stream test --resolution 1920x1080
```

### 4.4 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| --server, -s | MediaMTX server address | localhost |
| --stream, -n | Stream name/path | test |
| --protocol, -p | Streaming protocol: rtmp, rtsp, hls, srt | rtmp |
| --resolution, -r | Window resolution (e.g., 1920x1080) | Auto-detect |
| --fullscreen, -f | Start in fullscreen mode | Off |
| --volume, -v | Initial volume level 0-100 | 100 |
| --stats | Show statistics overlay | Off |
| --buffer | Frame buffer size (lower = less latency) | 5 |

### 4.5 Keyboard Controls

| Key | Action |
|-----|--------|
| **F** | Toggle fullscreen mode |
| **R** | Reconnect to the stream |
| **S** | Toggle statistics overlay (FPS, resolution, connection status) |
| **+** | Increase volume |
| **-** | Decrease volume |
| **M** | Mute / unmute audio |
| **Q** | Quit the player |
| **Esc** | Quit the player |

### 4.6 Auto-Start on Boot

#### Linux (systemd)

Create `/etc/systemd/system/corporate-tv.service`:

```ini
[Unit]
Description=Corporate TV Player
After=network.target

[Service]
Type=simple
User=YOUR_USER
Environment=DISPLAY=:0
ExecStart=/usr/bin/python3 /path/to/IPTV_Player_v2.py \
  --server YOUR_SERVER_IP --stream YOUR_STREAM --fullscreen
Restart=always
RestartSec=5

[Install]
WantedBy=graphical.target
```

Enable and start the service:

```bash
sudo systemctl enable corporate-tv
sudo systemctl start corporate-tv
```

#### macOS (launchd)

Create `~/Library/LaunchAgents/com.corporate.tvplayer.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.corporate.tvplayer</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/path/to/IPTV_Player_v2.py</string>
    <string>--server</string>
    <string>YOUR_SERVER_IP</string>
    <string>--stream</string>
    <string>YOUR_STREAM</string>
    <string>--fullscreen</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.corporate.tvplayer.plist
```

#### Windows (Auto-Start with Power Recovery)

For Windows devices that need to automatically start the player after a power outage, follow these steps to configure both Windows auto-login and automatic application startup.

**Step 1: Configure BIOS/UEFI for Power Recovery**

Configure the computer to automatically turn on when power is restored:

- Restart the computer and enter BIOS/UEFI (usually by pressing F2, F10, DEL, or ESC during boot)
- Navigate to Power Management or Advanced settings
- Find 'AC Power Recovery', 'After Power Loss', or 'Restore on AC Power Loss'
- Set it to 'Power On' or 'Last State'
- Save and exit BIOS

> **Note:** The exact setting name varies by manufacturer. Common names: 'AC Power Recovery' (Dell), 'Restore on AC Power Loss' (HP), 'After Power Failure' (Intel NUC).

**Step 2: Configure Windows Auto-Login**

Enable automatic login so Windows boots directly to the desktop:

*Method A: Using netplwiz (Recommended)*

- Press Win + R, type 'netplwiz' and press Enter
- Uncheck 'Users must enter a user name and password to use this computer'
- Click Apply
- Enter the username and password when prompted
- Click OK

*Method B: Using Registry (if Method A checkbox is not visible)*

Open Registry Editor (regedit) and navigate to:

```
HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon
```

Create or modify these string values:

```
AutoAdminLogon = 1
DefaultUserName = YOUR_USERNAME
DefaultPassword = YOUR_PASSWORD
DefaultDomainName = YOUR_COMPUTER_NAME
```

> **Important:** For security, consider using a dedicated local account with limited privileges for the TV display.

**Step 3: Create a Startup Batch Script**

Create a batch file to launch the player. Save as `C:\CorporateTV\start_player.bat`:

```batch
@echo off
REM Corporate TV Player Startup Script
REM Wait for network to be available
timeout /t 15 /nobreak

REM Change to the player directory
cd /d C:\CorporateTV

REM Start the player
:start
python IPTV_Player_v2.py --server YOUR_SERVER_IP --stream YOUR_STREAM --fullscreen

REM If player exits, wait and restart
timeout /t 5 /nobreak
goto start
```

**Step 4: Add to Windows Startup**

*Option A: Startup Folder (Simplest)*

- Press Win + R, type 'shell:startup' and press Enter
- Create a shortcut to `start_player.bat` in this folder
- Right-click the shortcut > Properties > set 'Run' to 'Minimized'

*Option B: Task Scheduler (More Reliable)*

- Open Task Scheduler (taskschd.msc)
- Click 'Create Task' (not Basic Task)
- General tab: Name it 'Corporate TV Player', check 'Run whether user is logged on or not'
- Triggers tab: Click New > 'At startup' and also add 'At log on'
- Actions tab: Click New > Action: 'Start a program' > Browse to `start_player.bat`
- Conditions tab: Uncheck 'Start only if on AC power'
- Settings tab: Check 'If the task fails, restart every: 1 minute'
- Click OK and enter credentials when prompted

**Step 5: Configure Windows Power Settings**

Prevent Windows from sleeping or turning off the display:

- Open Settings > System > Power & sleep
- Set 'Screen' to Never (when plugged in)
- Set 'Sleep' to Never (when plugged in)

Or use PowerShell (run as Administrator):

```powershell
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

**Step 6: Disable Windows Update Restarts (Optional)**

Prevent Windows from restarting during active hours:

- Open Settings > Windows Update > Advanced options
- Set Active hours to cover your display hours (e.g., 6:00 AM to 11:00 PM)
- Or use Group Policy (gpedit.msc) to configure update behavior

**Step 7: Hide Desktop Icons and Taskbar (Optional)**

For a cleaner display before the player starts:

- Right-click desktop > View > uncheck 'Show desktop icons'
- Right-click taskbar > Taskbar settings > turn on 'Automatically hide the taskbar'
- Set a solid black desktop background

> **Tip:** Test the complete power recovery cycle by unplugging the device and plugging it back in. The system should: power on automatically, log in to Windows, and start the player.

#### Windows Complete Setup Checklist

| Step | Action | Verified |
|------|--------|----------|
| 1 | BIOS: AC Power Recovery set to 'Power On' | ☐ |
| 2 | Windows: Auto-login configured | ☐ |
| 3 | Startup script created (start_player.bat) | ☐ |
| 4 | Script added to Startup folder or Task Scheduler | ☐ |
| 5 | Power settings: Display and sleep set to 'Never' | ☐ |
| 6 | Network: Static IP or DHCP reservation configured | ☐ |
| 7 | Test: Power cycle completes successfully | ☐ |

---

## 5. Browser-Based Playback (WebRTC and HLS)

MediaMTX provides built-in web players for both WebRTC (ultra-low latency) and HLS (high compatibility) streaming directly in the browser.

### 5.1 WebRTC Playback (Port 8889)

WebRTC provides the lowest latency (~0.5 seconds) for browser playback:

```
http://SERVER_IP:8889/your-stream-name
```

> **Important:** WebRTC requires Opus audio codec. If your stream uses AAC (standard from RTMP), you will have video but NO audio. See the WebRTC Audio Fix section below.

### 5.2 HLS Playback (Port 8888)

HLS provides excellent compatibility with 3-5 seconds latency:

```
http://SERVER_IP:8888/your-stream-name
```

> **Tip:** HLS has higher latency but supports AAC audio natively. Use this if you need audio without additional configuration.

### 5.3 Protocol Comparison

| Aspect | WebRTC | HLS |
|--------|--------|-----|
| Port | 8889 | 8888 |
| Latency | ~0.5-1 second | ~3-5 seconds |
| Audio from RTMP | No (needs Opus) | Yes (AAC supported) |
| Browser Support | Good | Excellent |
| Best For | Low latency, no audio | Compatibility, audio needed |

### 5.4 WebRTC Audio Fix

When streaming from OBS to MediaMTX using RTMP, WebRTC playback has video but no audio. This is a codec compatibility issue: OBS sends AAC audio, but WebRTC browsers only support Opus.

#### Solution 1: Use WHIP from OBS (Recommended)

If you have OBS Studio 30.0 or higher, use WHIP for native Opus audio:

- Go to Settings > Stream
- Set Service to WHIP
- Set Server to: `http://localhost:8889/your-stream/whip`
- Start streaming - both video AND audio will work in WebRTC

#### Solution 2: Use HLS Instead

Simply change the port from 8889 to 8888:

| Protocol | URL | Audio |
|----------|-----|-------|
| WebRTC | http://localhost:8889/your-stream | No (AAC not supported) |
| HLS | http://localhost:8888/your-stream | Yes (AAC supported) |

#### Solution 3: FFmpeg Audio Transcoding

Transcode AAC to Opus using FFmpeg:

```bash
ffmpeg -i rtmp://localhost:1935/your-stream \
  -c:v copy \
  -c:a libopus -b:a 128k \
  -f rtsp rtsp://localhost:8554/your-stream-webrtc
```

Then access the transcoded stream:

```
http://localhost:8889/your-stream-webrtc
```

#### Solution 4: Use the Python Player

The Corporate TV Player handles audio via FFplay, which supports AAC natively:

```bash
python IPTV_Player_v2.py --server localhost --stream your-stream
```

### 5.5 Decision Guide

| Scenario | Recommended Solution |
|----------|---------------------|
| Need browser playback with low latency + audio | Use WHIP from OBS 30+ |
| Need browser playback, latency not critical | Use HLS (port 8888) |
| Dedicated display devices (NUC, Mac Mini) | Use Python Player |
| Must keep RTMP workflow, need WebRTC audio | Use FFmpeg transcoding |

---

## 6. Troubleshooting Guide

### 6.1 MediaMTX Issues

| Problem | Solution |
|---------|----------|
| Port already in use error | Check what's using the port:<br>`lsof -i :1935` (macOS/Linux)<br>`netstat -ano \| findstr 1935` (Windows)<br>Kill the process or change the port in mediamtx.yml. |
| Firewall blocking connections | macOS: System Settings > Privacy & Security > Open Anyway<br>Linux: `sudo ufw allow 1935/tcp 8554/tcp 8888/tcp 8889/tcp 9997/tcp` |
| Stream shows in API but won't play | Check 'ready' field is true in API response.<br>Verify 'tracks' contains H264.<br>Try different protocol. |
| WebRTC shows 'Poor Connection' | Use x264 encoder (not hardware).<br>Set Profile to 'baseline'.<br>Try HLS instead. |
| RTSP not starting | Check `rtspAddress: :8554` in mediamtx.yml.<br>Verify no YAML syntax errors. |

### 6.2 OBS Studio Issues

| Problem | Solution |
|---------|----------|
| Failed to connect to server | Verify MediaMTX is running.<br>Check Server URL format: `rtmp://IP:1935` (no trailing slash).<br>Ensure firewall allows port 1935. |
| High latency/delay | Reduce Keyframe Interval to 1-2 seconds.<br>Enable 'zerolatency' tune.<br>Use RTMP protocol. |
| Choppy/stuttering video | Reduce bitrate to 2000 Kbps.<br>Reduce resolution to 720p.<br>Use 'veryfast' CPU preset. |
| Player shows black screen | Verify stream is active at `http://SERVER:9997/v3/paths/list`<br>Ensure Stream Key matches.<br>Check encoder is x264. |
| No audio on playback | Check Audio Mixer shows activity.<br>Verify AAC encoder is selected.<br>Ensure audio source is not muted. |

### 6.3 Corporate TV Player Issues

| Problem | Solution |
|---------|----------|
| No video (black screen) | Check stream is active at `http://SERVER:9997/v3/paths/list`<br>Try `--protocol hls`<br>Verify FFmpeg is installed. |
| Cannot connect to server | Verify MediaMTX is running.<br>Check firewall allows required ports.<br>Test: `ping SERVER_IP` |
| Audio not working | Verify FFmpeg is installed: `ffmpeg -version`<br>Check system volume.<br>Press M to unmute in player. |
| Choppy/stuttering playback | Increase buffer: `--buffer 10`<br>Try HLS: `--protocol hls`<br>Use wired Ethernet. |
| High latency | Decrease buffer: `--buffer 2`<br>Use RTMP or RTSP.<br>Set OBS keyframe interval to 1-2 seconds. |
| Screen resolution issues | Install screeninfo: `pip install screeninfo`<br>Specify manually: `--resolution 1920x1080` |

### 6.4 WebRTC Audio Issues

| Problem | Solution |
|---------|----------|
| Video plays but no audio | WebRTC requires Opus codec.<br>Use WHIP from OBS 30+, or use HLS (port 8888), or transcode with FFmpeg. |
| AAC audio with RTMP source | OBS sends AAC over RTMP. WebRTC browsers only support Opus.<br>Switch to WHIP or HLS. |

---

## 7. Quick Reference

### 7.1 Startup Checklist

| # | Action | Command / Details |
|---|--------|-------------------|
| 1 | Start MediaMTX | `mediamtx` (or run the executable) |
| 2 | Start OBS Streaming | OBS > Start Streaming to rtmp://SERVER:1935 |
| 3 | Verify Stream Active | http://SERVER:9997/v3/paths/list |
| 4 | Start Clients | Python player, browser, or VLC |

### 7.2 Default Ports

| Protocol | Port | URL Pattern |
|----------|------|-------------|
| RTMP | 1935 | rtmp://SERVER:1935/stream-name |
| RTSP | 8554 | rtsp://SERVER:8554/stream-name |
| HLS | 8888 | http://SERVER:8888/stream-name |
| WebRTC | 8889 | http://SERVER:8889/stream-name |
| SRT | 8890 | srt://SERVER:8890?streamid=stream-name |
| API | 9997 | http://SERVER:9997/v3/paths/list |

### 7.3 OBS Recommended Settings

| Setting | Value |
|---------|-------|
| Service | Custom... |
| Server | rtmp://SERVER_IP:1935 |
| Encoder | x264 |
| Rate Control | CBR |
| Bitrate | 2500 Kbps |
| Keyframe Interval | 2 seconds |
| CPU Preset | veryfast |
| Profile | baseline |
| Tune | zerolatency |

### 7.4 Player Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F | Toggle fullscreen |
| R | Reconnect |
| S | Toggle stats overlay |
| +/- | Volume up/down |
| M | Mute/unmute |
| Q / Esc | Quit |

### 7.5 Protocol Selection Guide

| Use Case | Protocol | Port |
|----------|----------|------|
| OBS to MediaMTX (input) | RTMP | 1935 |
| Python Player | RTMP or RTSP | 1935 / 8554 |
| Browser (low latency) | WebRTC | 8889 |
| Browser (compatibility) | HLS | 8888 |
| VLC Playback | RTSP or RTMP | 8554 / 1935 |
| Unreliable Network | SRT | 8890 |

---

## 8. Document Version History

This section documents the changes made in each version of this guide.

### Version 1.0 - Initial Release

*Release Date: March 2026*

Initial documentation consolidating multiple source documents into a single comprehensive guide.

**Contents:**

- System Architecture overview with component diagram
- MediaMTX Server installation and configuration (macOS, Linux, Docker)
- OBS Studio streaming configuration and recommended settings
- Corporate TV Player (Python/OpenCV) installation and usage
- Browser-based playback via WebRTC and HLS
- WebRTC audio fix solutions (WHIP, HLS, FFmpeg transcoding)
- Troubleshooting guide for common issues
- Quick reference tables for ports, settings, and keyboard shortcuts

### Version 1.1 - Windows Client Auto-Start

*Release Date: March 2026*

Added comprehensive Windows auto-start configuration for the Corporate TV Player client.

**New Content:**

- BIOS/UEFI configuration for AC Power Recovery
- Windows auto-login configuration (netplwiz and Registry methods)
- Startup batch script with auto-restart capability
- Task Scheduler configuration for reliable auto-start
- Windows power settings to prevent sleep and display timeout
- Optional display cleanup (hide icons, taskbar)
- Complete Windows setup verification checklist

**Fixes:**

- Corrected code block formatting - each line now displays on separate lines
- Added proper architecture diagram image embedded in document

### Version 1.2 - Server and OBS Auto-Start

*Release Date: March 2026*

Added auto-start configurations for MediaMTX server and OBS Studio to enable complete system recovery after power outages.

**New Content - MediaMTX Auto-Start:**

- Windows service installation using NSSM (Non-Sucking Service Manager)
- Service configuration with automatic restart on failure
- Logging configuration for Windows MediaMTX service
- Alternative Task Scheduler method for Windows
- Complete service management commands

**New Content - OBS Studio Auto-Start:**

- OBS command line parameters for automation (--startstreaming, --minimize-to-tray)
- Windows batch script for OBS auto-start with monitoring and auto-restart
- macOS launchd configuration for OBS auto-streaming
- Linux systemd user service for OBS auto-streaming
- Task Scheduler configuration for Windows OBS auto-start
- Important considerations for production OBS auto-start deployments

**New Content - Documentation:**

- Added Document Version History section (this section)

### Summary of Auto-Start Capabilities by Version

| Component | v1.0 | v1.1 | v1.2 |
|-----------|------|------|------|
| MediaMTX (Linux) | ✓ systemd | ✓ systemd | ✓ systemd |
| MediaMTX (macOS) | ✓ launchd | ✓ launchd | ✓ launchd |
| MediaMTX (Windows) | — | — | ✓ NSSM/Task Scheduler |
| OBS Studio (Windows) | — | — | ✓ Script + Task Scheduler |
| OBS Studio (macOS) | — | — | ✓ launchd |
| OBS Studio (Linux) | — | — | ✓ systemd user service |
| Client Player (Linux) | ✓ systemd | ✓ systemd | ✓ systemd |
| Client Player (macOS) | ✓ launchd | ✓ launchd | ✓ launchd |
| Client Player (Windows) | — | ✓ Full setup | ✓ Full setup |

> **Tip:** With version 1.2, the complete streaming pipeline (MediaMTX server, OBS source, and client players) can automatically recover from power outages on all major platforms.

---

*End of Document*