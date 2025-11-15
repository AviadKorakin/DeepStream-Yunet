# DeepStream YuNet Face Detector (Jetson + x86)

This project is a **minimal YuNet-based face detector** built on **NVIDIA DeepStream** and `nvinfer`, with:

- YuNet running as a DeepStream primary GIE (PGIE) via TensorRT.
- **5 facial keypoints** decoded per face (eyes, nose, mouth corners).
- End–to–end pipeline: **video → detection → OSD → MP4**.
- Verified to reach **~72 FPS on Jetson Orin Nano** (YuNet + 5 kpts, 640×640).

---

## Features

- DeepStream 7.x style pipeline using:

  - `uridecodebin` → `nvstreammux` → `nvinfer` (YuNet) → `nvdsosd` → encoder → MP4.

- YuNet face detection with:

  - Bounding boxes.
  - 5 facial keypoints (per face).

- Supports **file URIs** (`file:///…`) and can be extended to RTSP/HTTP easily.
- Works on **Jetson** (Orin Nano, Xavier, etc.) and **x86 with NVIDIA GPU**.

---

## Requirements

- NVIDIA **Jetson** with JetPack + DeepStream installed  
  or x86 machine with:

  - NVIDIA GPU + drivers
  - CUDA
  - DeepStream SDK

- Python 3
- DeepStream Python bindings (`pyds`) available, usually via:

```bash
  /opt/nvidia/deepstream/deepstream/lib
```

---

## Setup

Before running the example, run the setup script once:

```bash
./setup.sh
```

Typical things `setup.sh` would handle (depending on how you implemented it):

- Creating/activating a Python venv.
- Installing required Python packages (e.g. `gi`, `numpy`).
- Exporting `PYTHONPATH` or `LD_LIBRARY_PATH` so that `pyds` and GStreamer plugins are found.
- Any DeepStream / YuNet engine build steps you need.

> **Note:** Adjust `setup.sh` to your environment (Jetson vs PC, DeepStream version, etc.).

---

## YuNet nvinfer Config

You need a YuNet `nvinfer` config file, for example:

```text
  /home/swish/yunet/yunet_nvinfer_config.txt
```

This config should:

- Point to your YuNet TensorRT engine or ONNX model.
- Set the correct network input size (e.g. 640×640).
- Enable tensor output if you’re using a custom parser or Python tensor decode:

```ini
  [property]
  output-tensor-meta=1
```

(Refer to your DeepStream docs / existing config for full details.)

---

## Running the Example

Basic example command (your working reference):

```bash
python3 deepstream_yunet.py \
  -s file:///home/swish/yunet/example.mp4 \
  -c /home/swish/yunet/yunet_nvinfer_config.txt \
  --out out/yunet_ds_out.mp4 \
  --streammux-width 640 \
  --streammux-height 640 \
  --streammux-batch-size 1 \
  --bitrate 6000000 \
  --gop 30
```

### Arguments Explained

- `-s / --source`
  URI of the input video. For local files, use `file:///absolute/path/to/video.mp4`.

- `-c / --infer-config`
  Path to the `nvinfer` config file for YuNet.

- `--out`
  Output MP4 path for the encoded result (with drawn face boxes + keypoints).

- `--streammux-width`, `--streammux-height`
  Output resolution of `nvstreammux`. In the example: **640×640**
  (this is also a good match for YuNet input size).

- `--streammux-batch-size`
  Batch size for `nvstreammux`. For a single source, keep it at `1`.

- `--bitrate`
  Encoder target bitrate (here: `6,000,000` ≈ 6 Mbps).

- `--gop`
  GOP / keyframe interval. `30` means one I-frame every 30 frames.

---

## Demo Video Output

After a successful run, the DeepStream + YuNet pipeline writes the annotated video to:

```text
  out/yunet_ds_out.mp4
```

Open this file in your favorite video player (e.g. `vlc`, `mpv`, or a browser player) to see:

- YuNet face bounding boxes.
- 5 keypoints per face (eyes, nose, mouth corners).
- Real-time FPS overlay in the logs (see console output).

Example:

```bash
vlc out/yunet_ds_out.mp4
# or
mpv out/yunet_ds_out.mp4
```

---

## Expected Performance

On **Jetson Orin Nano** (properly configured, max clocks, DeepStream installed) this pipeline has been observed to reach:

- ~**72 FPS** at 640×640,
- YuNet + 5 keypoints per face,
- writing MP4 to disk.

Actual FPS will depend on:

- Input resolution and frame rate.
- Number of faces per frame.
- GPU / power mode / `jetson_clocks` state.
- Whether you enable extra logging, tensor printing, etc.

---

## Quick Start Summary

1. Install DeepStream + dependencies (Jetson or x86).

2. Run:

```bash
   ./setup.sh
```

3. Run the example:

```bash
   python3 deepstream_yunet.py \
     -s file:///home/swish/yunet/example.mp4 \
     -c /home/swish/yunet/yunet_nvinfer_config.txt \
     --out out/yunet_ds_out.mp4 \
     --streammux-width 640 \
     --streammux-height 640 \
     --streammux-batch-size 1 \
     --bitrate 6000000 \
     --gop 30
```

4. Inspect `out/yunet_ds_out.mp4` to see YuNet detections + 5 keypoints drawn at high FPS.

## References

- YuNet model & demos (OpenCV Zoo):
  [https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Original training code (libfacedetection / YuNet training):
  [https://github.com/ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
