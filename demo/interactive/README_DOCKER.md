# Docker Setup for KTO Lunar Lander Demo

This guide explains how to run the KTO Lunar Lander demo in Docker with GUI support.

## Prerequisites

- Docker Desktop installed
- X Server for display forwarding (OS-specific)

## X Server Setup (Required for GUI)

### Windows

1. **Install VcXsrv** (X Server for Windows):
   - Download from: https://sourceforge.net/projects/vcxsrv/
   - Install with default settings

2. **Configure VcXsrv**:
   - Launch XLaunch from Start Menu
   - Select "Multiple windows"
   - Display number: `0` (default)
   - **IMPORTANT**: Check "Disable access control" on the last screen
   - Click "Finish"

3. **Allow through Windows Firewall** (if prompted):
   - Allow VcXsrv through both Private and Public networks

### Linux

X11 is usually pre-installed. Just allow Docker to access display:

```bash
xhost +local:docker
```

### macOS

1. **Install XQuartz**:
   ```bash
   brew install --cask xquartz
   ```

2. **Configure XQuartz**:
   - Launch XQuartz
   - Go to Preferences → Security
   - Check "Allow connections from network clients"
   - Restart XQuartz

3. **Allow connections**:
   ```bash
   xhost +localhost
   ```

## Quick Start

### Option 1: Docker Compose (Recommended)

**Linux/macOS:**
```bash
cd demo/interactive
docker-compose up --build
```

**Windows:**
```bash
cd demo/interactive
docker-compose --profile windows up --build
```

### Option 2: Manual Docker Commands

**Build:**
```bash
cd demo/interactive
docker build -t lunar-lander-kto -f Dockerfile ../..
```

**Run (Windows):**
```bash
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  -v "%cd%/../../:/app" \
  -v "%cd%/recorded_episodes:/app/demo/interactive/recorded_episodes" \
  lunar-lander-kto
```

**Run (Linux/macOS):**
```bash
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v "$(pwd)/../../:/app" \
  -v "$(pwd)/recorded_episodes:/app/demo/interactive/recorded_episodes" \
  --network host \
  lunar-lander-kto
```

## Verifying X11 Connection

Test X11 forwarding before running the demo:

```bash
# Test with xeyes
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  lunar-lander-kto \
  bash -c "apt-get update && apt-get install -y x11-apps && xeyes"
```

If `xeyes` opens a window, X11 is working correctly.

## Controls

Once the game window opens:

- **↑ Arrow**: Main engine (Teleop mode)
- **← → Arrows**: Rotate left/right (Teleop mode)
- **R**: Reset episode
- **Q/Escape**: Quit
- **1**: Teleop mode (human control)
- **2**: Heuristic mode (PID controller)
- **3**: KTO mode (Drake trajectory optimization)
- **F**: Cycle failure level (Heuristic mode)
- **E**: Toggle episode recording

## Troubleshooting

### Issue: "cannot open display"

**Cause**: X Server not running or Docker can't connect.

**Solutions**:
- Ensure VcXsrv/XQuartz is running
- Windows: Check "Disable access control" in VcXsrv
- Linux: Run `xhost +local:docker`
- macOS: Run `xhost +localhost`

### Issue: "Connection refused to localhost:0"

**Windows users**: Use `host.docker.internal:0` instead of `localhost:0`

### Issue: Drake import fails

**Cause**: Wrong Python version in container.

**Solution**: Rebuild with `--no-cache`:
```bash
docker build --no-cache -t lunar-lander-kto -f Dockerfile ../..
```

### Issue: Recorded episodes not persisting

**Cause**: Volume mount not working.

**Solution**: Check volume mount paths in docker-compose.yml or run command.

## Development Workflow

The code is mounted as a volume, so changes are reflected immediately:

1. Edit code on host machine
2. Restart container: `docker-compose restart`
3. Or exec into running container: `docker exec -it lunar-lander-kto bash`

## Accessing Recorded Episodes

Episodes are saved to `./recorded_episodes/` on the host machine.

View episode statistics:
```bash
# Inside container
python -c "
import json
with open('recorded_episodes/metadata.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

## GPU Support (Optional)

For GPU-accelerated diffusion model training, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires: NVIDIA Docker runtime installed.

## Performance Notes

- **Drake planning**: ~1-2 seconds per episode (CPU-bound)
- **Pygame rendering**: 50 FPS target
- **Container overhead**: Negligible (<5%)

## Cleaning Up

Remove container and image:
```bash
docker-compose down
docker rmi lunar-lander-kto
```

Clean build cache:
```bash
docker builder prune
```
