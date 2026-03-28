# Corporate Network Configurations for IPTV
## Enterprise Network Configuration Guide
### Cisco ISE, Zscaler, QoS, and Firewall Configuration

**Version 1.0**  
**March 2026**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Port Requirements and Firewall Rules](#3-port-requirements-and-firewall-rules)
4. [Bandwidth Planning and QoS Configuration](#4-bandwidth-planning-and-qos-configuration)
5. [Cisco ISE Configuration](#5-cisco-ise-configuration)
6. [Zscaler Configuration](#6-zscaler-configuration)
7. [LAN vs WAN Deployment Options](#7-lan-vs-wan-deployment-options)
8. [Browser and Mobile Access](#8-browser-and-mobile-access)
9. [Network Segmentation and VLANs](#9-network-segmentation-and-vlans)
10. [Monitoring and Troubleshooting](#10-monitoring-and-troubleshooting)
11. [Implementation Checklist](#11-implementation-checklist)

---

## 1. Executive Summary

This guide provides comprehensive network configuration requirements for deploying the Corporate TV IPTV streaming system in an enterprise environment. The system uses MediaMTX as the streaming server, OBS Studio for content production, and supports multiple playback protocols including WebRTC (low-latency), HLS (broad compatibility), RTMP, and RTSP.

Key considerations covered in this guide include: firewall and port requirements, QoS marking for video traffic prioritization, Cisco ISE device profiling and authorization, Zscaler bypass configuration for internal streaming traffic, and deployment options for LAN-only versus WAN-extended scenarios.

### Critical Success Factors

| Factor | Requirement |
|--------|-------------|
| **Zscaler Bypass** | Internal streaming traffic MUST bypass Zscaler to avoid latency and throttling |
| **QoS Marking** | Video traffic should be marked AF41 (DSCP 34) for proper prioritization |
| **UDP for WebRTC** | WebRTC requires UDP ports for optimal low-latency performance |
| **Bandwidth Planning** | 1080p streaming requires approximately 3-5 Mbps per viewer |

---

## 2. System Architecture Overview

The Corporate TV system follows a hub-and-spoke architecture with the MediaMTX server at the center. OBS Studio ingests content via RTMP, and MediaMTX transcodes/distributes to viewers using multiple protocols.

### Component Data Flow

```
┌─────────────────┐     RTMP (TCP 1935)     ┌─────────────────┐
│   OBS Studio    │ ──────────────────────► │  MediaMTX       │
│   (Producer)    │                         │  Server         │
└─────────────────┘                         └────────┬────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    │                                │                                │
                    ▼                                ▼                                ▼
        ┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
        │ WebRTC (UDP 8889) │          │ HLS (TCP 8888)    │          │ RTSP (TCP 8554)   │
        │ < 0.5s latency    │          │ 5-15s latency     │          │ 1-2s latency      │
        │ Browser clients   │          │ Mobile/Browser    │          │ Python/VLC        │
        └───────────────────┘          └───────────────────┘          └───────────────────┘
```

### Protocol Comparison

| Protocol | Latency | Port(s) | Best For | Firewall Complexity |
|----------|---------|---------|----------|---------------------|
| WebRTC | < 0.5s | 8889 + UDP range | Real-time viewing | Complex (UDP) |
| HLS | 5-15s | 8888 (TCP) | Mobile, browsers | Simple (HTTP) |
| RTSP | 1-2s | 8554 (TCP) | Dedicated clients | Moderate |
| RTMP | 1-3s | 1935 (TCP) | Ingest only | Simple |

---

## 3. Port Requirements and Firewall Rules

### Required Ports Matrix

| Port | Protocol | Service | Direction | Required |
|------|----------|---------|-----------|----------|
| `1935` | TCP | RTMP (OBS Ingest) | OBS → Server | Yes |
| `8554` | TCP | RTSP Playback | Clients → Server | Optional |
| `8888` | TCP | HLS Playback (HTTP) | Clients → Server | Yes |
| `8889` | TCP+UDP | WebRTC Signaling | Clients ↔ Server | Yes |
| `8000-65535` | UDP | WebRTC RTP/RTCP | Clients ↔ Server | Yes* |
| `9997` | TCP | MediaMTX API | Admin → Server | Optional |

> **Note:** WebRTC UDP range can be constrained in `mediamtx.yml` using `webrtcAdditionalHosts` and port range settings.

### Cisco ASA/Firepower Firewall Rules

```cisco
! Allow RTMP ingest from OBS production network
access-list STREAMING permit tcp 10.100.0.0 255.255.255.0 host 10.50.0.10 eq 1935

! Allow HLS/WebRTC from user networks
access-list STREAMING permit tcp any host 10.50.0.10 eq 8888
access-list STREAMING permit tcp any host 10.50.0.10 eq 8889
access-list STREAMING permit udp any host 10.50.0.10 range 8000 65535

! Allow API access from management network
access-list STREAMING permit tcp 10.200.0.0 255.255.255.0 host 10.50.0.10 eq 9997
```

### WebRTC UDP Considerations

WebRTC uses ICE (Interactive Connectivity Establishment) which requires UDP connectivity for optimal performance. The wide port range (8000-65535) is needed for RTP/RTCP media streams. In restrictive firewall environments, WebRTC can fall back to TCP via TURN relay, but this increases latency significantly.

**Recommendations:**
- Ensure bidirectional UDP is allowed between clients and the MediaMTX server
- If UDP is completely blocked, fall back to HLS for playback
- Consider deploying an internal TURN server if NAT traversal is needed

---

## 4. Bandwidth Planning and QoS Configuration

### Bandwidth Requirements per Viewer

| Resolution | Bitrate | 50 Viewers | 100 Viewers | 250 Viewers |
|------------|---------|------------|-------------|-------------|
| 480p (SD) | 1.5 Mbps | 75 Mbps | 150 Mbps | 375 Mbps |
| 720p (HD) | 2.5 Mbps | 125 Mbps | 250 Mbps | 625 Mbps |
| 1080p (Full HD) | 4 Mbps | 200 Mbps | 400 Mbps | 1 Gbps |

> **Note:** Add 20% overhead for protocol headers and retransmissions.

### DSCP Marking Recommendations

Following Cisco best practices for video streaming:

| Traffic Type | DSCP Class | Decimal Value | Queue Treatment |
|--------------|------------|---------------|-----------------|
| Interactive Video | AF41 | 34 | Assured Forwarding, low drop |
| Streaming Video | CS4 | 32 | Class Selector 4 |
| Audio (if separate) | EF | 46 | Expedited Forwarding (priority) |
| Signaling | AF31 | 26 | Assured Forwarding, Class 3 |

### Cisco Catalyst Switch QoS Configuration

```cisco
! Create access list for streaming server
ip access-list extended STREAMING-SERVER
  permit ip any host 10.50.0.10
  permit ip host 10.50.0.10 any

! Create class maps for streaming traffic
class-map match-any CORPORATE-TV-VIDEO
  match access-group name STREAMING-SERVER
  match dscp af41 cs4

class-map match-any CORPORATE-TV-SIGNALING
  match dscp af31

! Create policy map
policy-map STREAMING-QOS
  class CORPORATE-TV-VIDEO
    bandwidth percent 30
    set dscp af41
  class CORPORATE-TV-SIGNALING
    bandwidth percent 5
    set dscp af31
  class class-default
    bandwidth percent 65
    fair-queue

! Apply to interface
interface GigabitEthernet1/0/1
  service-policy output STREAMING-QOS
```

### Wireless QoS (WMM) Configuration

For Cisco Wireless LAN Controller:

```cisco
! Configure QoS profile for streaming
config qos protocol-type platinum
config wlan qos corporate-tv platinum

! Enable WMM
config wlan wmm allow corporate-tv
```

---

## 5. Cisco ISE Configuration

### Device Profiling for Streaming Endpoints

Create a custom profiling policy in ISE to identify Corporate TV streaming devices (MediaMTX server, NUC players) and apply appropriate network access policies.

#### Step 1: Create Endpoint Identity Group

1. Navigate to **Administration > Identity Management > Groups > Endpoint Identity Groups**
2. Click **Add**
3. Name: `Corporate-TV-Devices`
4. Description: `MediaMTX servers and NUC streaming players`
5. Add MediaMTX server and NUC player MAC addresses to this group

#### Step 2: Create Authorization Profile

1. Navigate to **Policy > Policy Elements > Results > Authorization > Authorization Profiles**
2. Click **Add**
3. Configure:
   - Name: `Corporate-TV-Access`
   - Access Type: `ACCESS_ACCEPT`
   - VLAN: `50` (Streaming VLAN)
   - dACL: (optional) Create downloadable ACL for restricted access

#### Step 3: Create Authorization Policy Rule

1. Navigate to **Policy > Policy Sets > [Your Policy Set] > Authorization Policy**
2. Add new rule above default:

| Rule Name | Condition | Result |
|-----------|-----------|--------|
| Corporate-TV-Streaming-Access | EndPointGroup EQUALS Corporate-TV-Devices | Corporate-TV-Access |

### Downloadable ACL (Optional)

For restricting streaming device traffic:

```cisco
! dACL: Corporate-TV-Streaming-dACL
permit udp any any range 8000 65535
permit tcp any any eq 1935
permit tcp any any eq 8554
permit tcp any any eq 8888
permit tcp any any eq 8889
permit tcp any any eq 9997
permit icmp any any
deny ip any any
```

### QoS via ISE Authorization (Cisco AV-Pairs)

For wireless clients viewing Corporate TV, push QoS policies via ISE authorization:

**Authorization Profile Advanced Attributes:**

```
cisco-av-pair = ip:sub-qos-policy-in=STREAMING-QOS
cisco-av-pair = ip:sub-qos-policy-out=STREAMING-QOS
```

---

## 6. Zscaler Configuration

> ⚠️ **CRITICAL:** Internal streaming traffic MUST bypass Zscaler to avoid latency, throttling, and potential SSL inspection issues. Routing video through Zscaler will degrade playback quality significantly.

### Option 1: PAC File Bypass (Z-Tunnel 1.0)

For environments using Z-Tunnel 1.0, add bypasses to the PAC file:

```javascript
function FindProxyForURL(url, host) {
  // Bypass Corporate TV streaming server by IP
  if (isInNet(dnsResolve(host), "10.50.0.0", "255.255.255.0")) {
    return "DIRECT";
  }
  
  // Bypass Corporate TV by hostname
  if (dnsDomainIs(host, "corporatetv.company.local")) {
    return "DIRECT";
  }
  
  // Bypass streaming ports (if port-based routing available)
  if (shExpMatch(url, "*:8888/*") || 
      shExpMatch(url, "*:8889/*") ||
      shExpMatch(url, "*:1935/*")) {
    return "DIRECT";
  }
  
  // Default to Zscaler
  return "PROXY gateway.zscaler.net:443";
}
```

### Option 2: Destination Exclusions (Z-Tunnel 2.0)

For Z-Tunnel 2.0, configure destination exclusions in ZCC App Profile:

1. Navigate to **Zscaler Client Connector Portal > App Profiles**
2. Edit the relevant profile
3. Go to **Forwarding Profile > Destination Exclusions**
4. Add exclusions:

| Type | Value | Description |
|------|-------|-------------|
| IP Range | `10.50.0.0/24` | Streaming VLAN |
| IP Address | `10.50.0.10` | MediaMTX server |
| Hostname | `corporatetv.company.local` | Streaming DNS name |

### Option 3: IP-Based Application Bypass

Create an IP-based application bypass for streaming server:

1. Navigate to **ZCC Portal > Traffic Forwarding > App Bypass**
2. Click **Add Application**
3. Configure:
   - Application Name: `Corporate TV Streaming`
   - Type: `IP-Based`
   - Destination IP: `10.50.0.10`
   - Ports: `1935, 8554, 8888, 8889`
   - Action: `Bypass`

### Option 4: Zscaler Private Access (ZPA) for Remote Users

For remote/WAN users who need to access Corporate TV:

1. Create ZPA Application Segment:
   - Name: `Corporate TV`
   - Domain/IP: `10.50.0.10` or `corporatetv.company.local`
   - Ports: `8888, 8889`
   - Protocol: `TCP, UDP`

2. Create Access Policy allowing authorized users

### Verification

After configuring bypass, verify traffic flow:

```bash
# On client machine, check if traffic bypasses Zscaler
curl -v http://10.50.0.10:8888/corporate-tv/index.m3u8

# Check Zscaler logs - streaming traffic should NOT appear
# In ZIA Admin Portal: Logs > Web Insights
```

---

## 7. LAN vs WAN Deployment Options

### Decision Matrix

| Factor | LAN-Only | WAN/Hybrid |
|--------|----------|------------|
| **Latency** | < 0.5s (WebRTC) | 2-15s (HLS recommended) |
| **QoS Control** | Full control | Limited (ISP dependent) |
| **Zscaler Bypass** | Simple (internal only) | Complex (VPN/ZPA needed) |
| **Recommended Protocol** | WebRTC | HLS |
| **Bandwidth** | Shared on LAN backbone | WAN link constraint |
| **Firewall Complexity** | Internal rules only | External + VPN rules |
| **Reliability** | High | Variable (network dependent) |

### LAN-Only Deployment (Recommended for Single Site)

**Advantages:**
- Lowest latency with WebRTC (< 500ms)
- Full QoS control on Cisco infrastructure
- Simple Zscaler bypass via PAC or exclusions
- No WAN bandwidth concerns

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ OBS Studio  │────►│  MediaMTX   │────►│  Viewers    │
│ VLAN 100    │     │  VLAN 50    │     │  VLAN 10/20 │
└─────────────┘     └─────────────┘     └─────────────┘
        All traffic stays within campus network
```

### WAN/Multi-Site Deployment

**Considerations:**
- Use HLS for reliability over variable WAN links
- Accept higher latency (5-15 seconds) as tradeoff
- Consider edge caching or local MediaMTX instances at each site
- Use Zscaler Private Access (ZPA) for remote office access

**Architecture Options:**

**Option A: Centralized Server**
```
┌─────────────────────────────────────────────────────────┐
│                    Headquarters                          │
│  ┌──────────┐     ┌──────────┐                          │
│  │   OBS    │────►│ MediaMTX │                          │
│  └──────────┘     └────┬─────┘                          │
└────────────────────────┼────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
       ┌────────┐   ┌────────┐   ┌────────┐
       │Branch A│   │Branch B│   │ Remote │
       │  HLS   │   │  HLS   │   │  HLS   │
       └────────┘   └────────┘   └────────┘
```

**Option B: Distributed with Edge Caching**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     HQ       │     │   Branch A   │     │   Branch B   │
│ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │
│ │ MediaMTX │─┼────►│ │  Cache   │ │     │ │  Cache   │ │
│ │ (Origin) │ │     │ │  Server  │ │     │ │  Server  │ │
│ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 8. Browser and Mobile Access

### Protocol Support by Platform

| Platform | WebRTC | HLS | RTSP | Recommended |
|----------|--------|-----|------|-------------|
| Chrome/Edge (Desktop) | ✓ | ✓ | ✗ | WebRTC |
| Firefox | ✓ | ✓* | ✗ | WebRTC |
| Safari (macOS) | ✓ | ✓ | ✗ | HLS (native) |
| iOS Safari | ✓ | ✓ | ✗ | HLS (native) |
| Android Chrome | ✓ | ✓ | ✗ | WebRTC |
| VLC | ✓** | ✓ | ✓ | RTSP |

> \* Firefox requires hls.js library for HLS playback  
> \*\* VLC WebRTC support is limited

### Browser Playback URLs

**WebRTC (Low Latency):**
```
http://corporatetv.company.local:8889/corporate-tv
```

**HLS (Broad Compatibility):**
```
http://corporatetv.company.local:8888/corporate-tv/index.m3u8
```

**RTSP (VLC/Python Player):**
```
rtsp://corporatetv.company.local:8554/corporate-tv
```

### Embedding in Intranet Portal

**WebRTC Embed:**
```html
<iframe 
  src="http://corporatetv.company.local:8889/corporate-tv" 
  width="1280" 
  height="720" 
  frameborder="0" 
  allowfullscreen>
</iframe>
```

**HLS with hls.js:**
```html
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<video id="video" controls width="1280" height="720"></video>
<script>
  var video = document.getElementById('video');
  var videoSrc = 'http://corporatetv.company.local:8888/corporate-tv/index.m3u8';
  
  if (Hls.isSupported()) {
    var hls = new Hls();
    hls.loadSource(videoSrc);
    hls.attachMedia(video);
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    video.src = videoSrc;
  }
</script>
```

### WiFi Configuration for Mobile Viewers

Ensure corporate WiFi is configured for reliable streaming:

| Setting | Recommendation | Reason |
|---------|----------------|--------|
| WMM (WiFi Multimedia) | **Enable** | QoS prioritization for video |
| Band Steering | Prefer 5GHz | Higher bandwidth, less interference |
| Client Isolation | **Disable** | Clients need to reach streaming server |
| Airtime Fairness | Consider disabling | Prevents throttling of video clients |
| DTIM Interval | 1-3 | Better for real-time traffic |
| RTS/CTS | Enable if needed | Helps in high-density areas |

---

## 9. Network Segmentation and VLANs

### Recommended VLAN Architecture

| VLAN ID | Name | Subnet | Purpose |
|---------|------|--------|---------|
| 10 | Users-Wired | `10.10.0.0/16` | Wired workstations |
| 20 | WiFi-Corporate | `10.20.0.0/16` | Wireless clients |
| 30 | WiFi-Guest | `10.30.0.0/16` | Guest network (isolated) |
| 50 | Streaming | `10.50.0.0/24` | MediaMTX server |
| 100 | OBS-Production | `10.100.0.0/24` | OBS workstations |
| 200 | Management | `10.200.0.0/24` | Network management |

### VLAN Configuration (Cisco IOS)

```cisco
! Create VLANs
vlan 10
  name Users-Wired
vlan 20
  name WiFi-Corporate
vlan 50
  name Streaming
vlan 100
  name OBS-Production

! Configure SVI interfaces
interface Vlan10
  ip address 10.10.0.1 255.255.0.0
  description User Workstations
  
interface Vlan20
  ip address 10.20.0.1 255.255.0.0
  description WiFi Clients
  
interface Vlan50
  ip address 10.50.0.1 255.255.255.0
  description Streaming Infrastructure
  
interface Vlan100
  ip address 10.100.0.1 255.255.255.0
  description OBS Production
```

### Inter-VLAN Routing ACLs

```cisco
! Access list for streaming traffic control
ip access-list extended STREAMING-INTER-VLAN

! Allow OBS to reach MediaMTX (RTMP ingest)
permit tcp 10.100.0.0 0.0.0.255 10.50.0.0 0.0.0.255 eq 1935

! Allow Users (VLAN 10) to reach MediaMTX (playback)
permit tcp 10.10.0.0 0.0.255.255 10.50.0.0 0.0.0.255 eq 8888
permit tcp 10.10.0.0 0.0.255.255 10.50.0.0 0.0.0.255 eq 8889
permit udp 10.10.0.0 0.0.255.255 10.50.0.0 0.0.0.255 range 8000 65535

! Allow WiFi (VLAN 20) to reach MediaMTX (playback)
permit tcp 10.20.0.0 0.0.255.255 10.50.0.0 0.0.0.255 eq 8888
permit tcp 10.20.0.0 0.0.255.255 10.50.0.0 0.0.0.255 eq 8889
permit udp 10.20.0.0 0.0.255.255 10.50.0.0 0.0.0.255 range 8000 65535

! Allow Management access to API
permit tcp 10.200.0.0 0.0.0.255 10.50.0.0 0.0.0.255 eq 9997

! Deny direct guest access (optional - allow via explicit rule if needed)
deny ip 10.30.0.0 0.0.255.255 10.50.0.0 0.0.0.255

! Apply to VLAN interfaces
interface Vlan10
  ip access-group STREAMING-INTER-VLAN out
interface Vlan20
  ip access-group STREAMING-INTER-VLAN out
```

---

## 10. Monitoring and Troubleshooting

### MediaMTX API Monitoring

MediaMTX provides a REST API on port 9997 for monitoring:

```bash
# Check server configuration
curl http://10.50.0.10:9997/v3/config/global/get

# List all configured paths
curl http://10.50.0.10:9997/v3/paths/list

# Get specific path details
curl http://10.50.0.10:9997/v3/paths/get/corporate-tv

# List HLS muxers (active HLS streams)
curl http://10.50.0.10:9997/v3/hlsmuxers/list

# List WebRTC sessions
curl http://10.50.0.10:9997/v3/webrtcsessions/list

# List RTSP connections
curl http://10.50.0.10:9997/v3/rtspconns/list
```

### Network Diagnostic Commands

```bash
# Test TCP connectivity to HLS port
nc -zv 10.50.0.10 8888

# Test WebRTC signaling port
nc -zv 10.50.0.10 8889

# Test UDP connectivity (WebRTC media)
nc -u -zv 10.50.0.10 8889

# Verify HLS manifest accessibility
curl -I http://10.50.0.10:8888/corporate-tv/index.m3u8

# Check RTMP connectivity
nc -zv 10.50.0.10 1935

# Test from different VLANs
ping -c 4 10.50.0.10

# Check DSCP marking on packets
tcpdump -i eth0 -v host 10.50.0.10 | grep tos

# Capture streaming traffic for analysis
tcpdump -i eth0 -w streaming_capture.pcap host 10.50.0.10
```

### Cisco Switch Diagnostics

```cisco
! Verify QoS policy is applied
show policy-map interface GigabitEthernet1/0/1

! Check QoS statistics
show policy-map interface GigabitEthernet1/0/1 | include Class|packets

! Verify VLAN configuration
show vlan brief

! Check inter-VLAN routing
show ip route 10.50.0.0

! Verify ACL hits
show ip access-lists STREAMING-INTER-VLAN

! Check port status to streaming server
show interface status | include 10.50.0.10
```

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| WebRTC fails, HLS works | UDP blocked by firewall | Open UDP ports 8000-65535 to MediaMTX |
| High latency on WebRTC | Traffic routed through Zscaler | Add Zscaler bypass for streaming server |
| Choppy playback | Insufficient bandwidth | Apply QoS policies, check link saturation |
| Mobile playback fails | WiFi client isolation enabled | Disable client isolation on SSID |
| Intermittent buffering | Network congestion | Increase QoS bandwidth allocation |
| No audio in WebRTC | AAC codec issue | Configure Opus codec in OBS/MediaMTX |
| Can't reach server from VLAN | Missing inter-VLAN route/ACL | Check ACLs and routing table |
| HLS 404 errors | Stream not publishing | Verify OBS is connected and streaming |

### Log Locations

**MediaMTX Server:**
```bash
# If running as systemd service
journalctl -u mediamtx -f

# If running manually
# Logs output to stdout/stderr
```

**Cisco ISE:**
- Operations > RADIUS > Live Logs
- Operations > Reports > Endpoints and Users

**Zscaler:**
- ZIA Admin Portal > Logs > Web Insights
- ZCC Portal > Diagnostics > Connection Status

---

## 11. Implementation Checklist

### Phase 1: Network Preparation

- [ ] Create Streaming VLAN (e.g., VLAN 50)
- [ ] Assign IP subnet (e.g., 10.50.0.0/24)
- [ ] Configure SVI on core switch
- [ ] Configure inter-VLAN routing ACLs
- [ ] Open required ports on internal firewalls
- [ ] Create DNS record: `corporatetv.company.local` → `10.50.0.10`
- [ ] Verify DNS resolution from all client VLANs

### Phase 2: QoS Configuration

- [ ] Create QoS class maps on access switches
- [ ] Create QoS class maps on distribution switches
- [ ] Create and apply policy maps
- [ ] Configure WLC for WiFi QoS (WMM)
- [ ] Enable DSCP trust on access ports
- [ ] Verify DSCP marking with packet capture
- [ ] Test bandwidth allocation under load

### Phase 3: Cisco ISE Configuration

- [ ] Create endpoint identity group: `Corporate-TV-Devices`
- [ ] Add MediaMTX server MAC address
- [ ] Add NUC player MAC addresses
- [ ] Create authorization profile: `Corporate-TV-Access`
- [ ] Configure VLAN assignment
- [ ] Create downloadable ACL (optional)
- [ ] Create authorization policy rule
- [ ] Test authentication and authorization

### Phase 4: Zscaler Configuration

- [ ] Identify bypass method (PAC/exclusions/app bypass)
- [ ] Configure bypass for streaming server IP
- [ ] Configure bypass for streaming hostname
- [ ] Deploy updated PAC file or ZCC profile
- [ ] Test bypass with client-side packet capture
- [ ] Verify traffic does NOT traverse Zscaler cloud
- [ ] Configure ZPA for remote users (if needed)

### Phase 5: Server Deployment

- [ ] Deploy MediaMTX server to Streaming VLAN
- [ ] Assign static IP: 10.50.0.10
- [ ] Configure `mediamtx.yml` with appropriate ports
- [ ] Set up systemd service (Linux) or NSSM (Windows)
- [ ] Enable auto-start on boot
- [ ] Test API accessibility (port 9997)
- [ ] Configure firewall on server (if applicable)

### Phase 6: Client Testing

- [ ] Test RTMP ingest from OBS workstation
- [ ] Test WebRTC playback from wired workstation (VLAN 10)
- [ ] Test HLS playback from mobile device (VLAN 20)
- [ ] Test playback from different browser types
- [ ] Verify playback works from all source VLANs
- [ ] Measure and document latency metrics
- [ ] Test during peak network usage hours
- [ ] Verify QoS is functioning (packet capture)

### Phase 7: Documentation and Handoff

- [ ] Document final network configuration
- [ ] Document IP addresses and hostnames
- [ ] Create network diagram
- [ ] Create runbook for operations team
- [ ] Document troubleshooting procedures
- [ ] Set up monitoring alerts (API, connectivity)
- [ ] Train support staff on common issues
- [ ] Create end-user access guide
- [ ] Schedule periodic review of configurations

---

## Appendix A: Quick Reference

### Key IP Addresses

| Component | IP Address | Ports |
|-----------|------------|-------|
| MediaMTX Server | 10.50.0.10 | 1935, 8554, 8888, 8889, 9997 |
| OBS Workstation | 10.100.0.x | N/A (client) |
| Gateway (Streaming VLAN) | 10.50.0.1 | N/A |

### Key URLs

| Purpose | URL |
|---------|-----|
| WebRTC Playback | `http://corporatetv.company.local:8889/corporate-tv` |
| HLS Playback | `http://corporatetv.company.local:8888/corporate-tv/index.m3u8` |
| RTSP Playback | `rtsp://corporatetv.company.local:8554/corporate-tv` |
| API Status | `http://corporatetv.company.local:9997/v3/paths/list` |
| OBS RTMP Target | `rtmp://corporatetv.company.local:1935/corporate-tv` |

### DSCP Values

| Traffic | DSCP Class | Decimal | Binary |
|---------|------------|---------|--------|
| Video | AF41 | 34 | 100010 |
| Streaming | CS4 | 32 | 100000 |
| Audio | EF | 46 | 101110 |
| Signaling | AF31 | 26 | 011010 |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | March 2026 | Initial release |

---

*Document prepared for Corporate TV IPTV System deployment*