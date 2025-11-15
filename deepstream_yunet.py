#!/usr/bin/env python3
# DeepStream: YuNet → MP4, NVMM-safe linking, encoder/decoder fallbacks (Jetson + x86)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from threading import Lock
import ctypes
import math

# DeepStream Python bindings
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds  # noqa: E402

# ───────── Platform guards ─────────
IS_JETSON = platform.uname().machine == "aarch64"
NVMM_CAPS = "video/x-raw(memory:NVMM)"

# ───────── Global defaults (overridden by CLI) ─────────
SOURCE = ""  # URI (file:///..., rtsp://..., http(s)://...)
INFER_CONFIG = ""  # YuNet nvinfer config
OUT_MP4 = "yunet_out.mp4"

MUX_W = 1920  # nvstreammux width
MUX_H = 1080  # nvstreammux height
MUX_BATCH = 1  # nvstreammux batch-size

BITRATE = 8_000_000  # target encoder bitrate
GOP = 30  # keyframe interval
INSERT_SPS_PPS = True  # HW enc: insert SPS/PPS per keyframe

PERF_INTERVAL_SEC = 5  # how often to log FPS

# NEW: disable drawing (decode tensors + print only)
NO_DRAW = False
# Throttle tensor logging so we don't destroy FPS
TENSOR_LOG_EVERY = 30  # log every Nth frame per source

# YuNet strides
YUNET_STRIDES = [8, 16, 32]


# ───────── Utility: Jetson NVMM helper ─────────
def force_vic_surface_array(el):
    """
    Enable Jetson-specific zero-copy path; no-op on dGPU/x86.
    Safe to call for any nvvideoconvert / nv* element.
    """
    if not el:
        return el
    if IS_JETSON:
        try:
            el.set_property("copy-hw", "VIC")
        except Exception:
            pass
        try:
            el.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_SURFACE_ARRAY))
        except Exception:
            pass
    return el


# ───────── FPS tracker ─────────
class FPS:
    """Small helper to track instantaneous + average FPS per source."""

    def __init__(self, sid: int):
        self.sid = sid
        self.t0 = time.time()
        self.first = True
        self.fc = 0
        self.tsum = 0.0
        self.fsum = 0
        self._lock = Lock()

    def update(self):
        """Call once per frame for this source."""
        with self._lock:
            if self.first:
                self.t0 = time.time()
                self.first = False
                self.fc = 0
                self.tsum = 0.0
                self.fsum = 0
            else:
                self.fc += 1

    def snapshot(self):
        """
        Return (current_FPS, avg_FPS) and reset the short-term counter.
        Called by a GLib timer every PERF_INTERVAL_SEC.
        """
        with self._lock:
            now = time.time()
            dt = max(now - self.t0, 1e-6)
            self.tsum += dt
            self.fsum += self.fc
            cur = self.fc / dt
            avg = self.fsum / max(self.tsum, 1e-6)
            self.t0 = now
            self.fc = 0
        return cur, avg


# sid -> FPS tracker
perf = {}


# ───────── Tensor helpers (generic DeepStream best-practice) ─────────
def _get_layer(tmeta, i):
    """
    tmeta: NvDsInferTensorMeta
    return: NvDsInferLayerInfo for output i
    """
    f = getattr(pyds, "get_nvds_LayerInfo", None) or getattr(
        pyds, "get_nvds_layer_info", None
    )
    if not f:
        raise RuntimeError(
            "get_nvds_LayerInfo / get_nvds_layer_info not available in pyds"
        )
    return f(tmeta, i)


def _layer_name(layer):
    """
    Robustly extract layer name for various DeepStream versions.
    """
    attr = getattr(layer, "layerName", None) or getattr(layer, "name", None)
    try:
        return pyds.get_string(attr) if attr is not None else ""
    except Exception:
        return attr if isinstance(attr, str) else ""


def _dims(layer):
    """
    Return shape as a Python list.
    """
    return [int(layer.inferDims.d[j]) for j in range(layer.inferDims.numDims)]


def _to_np(layer):
    """
    Convert a NvDsInferLayerInfo buffer to a flat float32 numpy array,
    handling NVDS_HALF and NVDS_INT32 if needed.
    """
    import numpy as np

    dims = _dims(layer)
    numel = 1
    for d in dims:
        numel *= max(1, int(d))

    buf = pyds.get_ptr(layer.buffer)
    if not buf or numel <= 0:
        return None

    dt = getattr(layer, "dataType", None)
    D = getattr(pyds, "NvDsDataType", None)

    if D and dt == D.NVDS_HALF:
        raw = ctypes.string_at(buf, numel * 2)
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False)
    elif D and dt == D.NVDS_INT32:
        raw = ctypes.string_at(buf, numel * 4)
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        raw = ctypes.string_at(buf, numel * 4)
        arr = np.frombuffer(raw, dtype=np.float32)

    try:
        return arr.reshape(dims)
    except Exception:
        return arr


def _reshape_known(arr, comp):
    """
    Flatten to 1D and reshape to (?, comp) if possible.
    E.g. comp=4 for bbox, comp=10 for 5 keypoints.
    """
    import numpy as np

    if arr is None:
        return None
    a = arr
    while a.ndim > 2:
        a = a.reshape(a.shape[-2], a.shape[-1])
    a = a.reshape(-1)
    if a.size % comp != 0:
        return None
    j = a.size // comp
    return a.reshape(j, comp)


# ───────── YuNet decode in Python (anchor-free) ─────────
def _decode_yunet_stride_py(bbox, cls, obj, kps, stride, input_w, input_h, score_thr):
    """
    Vectorized YuNet per-stride decode.

    bbox: (N,4)
    cls:  (N,1) or (N,)
    obj:  (N,1) or (N,)
    kps:  (N,10)
    """
    import numpy as np

    dets = []
    if bbox is None or cls is None or obj is None or kps is None:
        return dets

    N = bbox.shape[0]
    cols = input_w // stride
    rows = input_h // stride
    if cols * rows != N or kps.shape[0] != N:
        return dets

    # Flatten to 1D
    cls = cls.reshape(-1)
    obj = obj.reshape(-1)

    # 1) scores in one shot
    cls_clamped = np.clip(cls, 0.0, 1.0)
    obj_clamped = np.clip(obj, 0.0, 1.0)
    scores = np.sqrt(cls_clamped * obj_clamped)

    # 2) filter by score threshold
    keep = scores >= score_thr
    if not np.any(keep):
        return dets

    scores_kept = scores[keep]
    bbox_kept = bbox[keep]  # (M,4)
    kps_kept = kps[keep]  # (M,10)

    # anchor indices -> row/col
    idxs = np.nonzero(keep)[0]
    r = idxs // cols
    c = idxs % cols

    r = r.astype(np.float32)
    c = c.astype(np.float32)

    dx = bbox_kept[:, 0].astype(np.float32)
    dy = bbox_kept[:, 1].astype(np.float32)
    dw = bbox_kept[:, 2].astype(np.float32)
    dh = bbox_kept[:, 3].astype(np.float32)

    # 3) decode centers + sizes (all at once)
    stride_f = float(stride)
    cx = (c + dx) * stride_f
    cy = (r + dy) * stride_f
    w = np.exp(dw) * stride_f
    h = np.exp(dh) * stride_f

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    # 4) clamp to input size
    x1 = np.clip(x1, 0.0, float(input_w - 1))
    y1 = np.clip(y1, 0.0, float(input_h - 1))
    x2 = np.clip(x2, 0.0, float(input_w - 1))
    y2 = np.clip(y2, 0.0, float(input_h - 1))

    # 5) landmarks (vectorized per kpt index)
    # kps_kept: (M,10) = [x0,y0,x1,y1,...]
    kps_arr = kps_kept.astype(np.float32)
    lm_list = []
    for n in range(5):
        lx_off = kps_arr[:, 2 * n + 0]
        ly_off = kps_arr[:, 2 * n + 1]
        lx = (lx_off + c) * stride_f
        ly = (ly_off + r) * stride_f
        lx = np.clip(lx, 0.0, float(input_w - 1))
        ly = np.clip(ly, 0.0, float(input_h - 1))
        lm_list.append((lx, ly))

    # 6) pack into list-of-dicts (only for kept anchors)
    #    This is now the only Python loop for this stride.
    M = scores_kept.shape[0]
    for i in range(M):
        lm = [(lm_list[n][0][i].item(), lm_list[n][1][i].item()) for n in range(5)]
        dets.append(
            {
                "bbox": (
                    x1[i].item(),
                    y1[i].item(),
                    x2[i].item(),
                    y2[i].item(),
                ),
                "score": float(scores_kept[i]),
                "landmarks": lm,
            }
        )

    return dets


def _nms_py(dets, nms_thr):
    """
    Greedy NMS over list of dicts: {'bbox': (x1,y1,x2,y2), 'score': s, ...}
    """
    if not dets:
        return []

    import numpy as np

    scores = np.array([d["score"] for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]

    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)

    for _i, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(dets[i])

        x1_i, y1_i, x2_i, y2_i = dets[i]["bbox"]
        area_i = max(0.0, x2_i - x1_i) * max(0.0, y2_i - y1_i)

        for j in order[_i + 1 :]:
            if suppressed[j]:
                continue
            x1_j, y1_j, x2_j, y2_j = dets[j]["bbox"]

            xx1 = max(x1_i, x1_j)
            yy1 = max(y1_i, y1_j)
            xx2 = min(x2_i, x2_j)
            yy2 = min(y2_i, y2_j)

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            area_j = max(0.0, x2_j - x1_j) * max(0.0, y2_j - y1_j)
            union = area_i + area_j - inter
            iou = inter / union if union > 0.0 else 0.0

            if iou > nms_thr:
                suppressed[j] = True

    return keep


def decode_yunet_from_tensor_meta(tmeta, score_thr=0.75, nms_thr=0.3):
    """
    Read YuNet heads from NvDsInferTensorMeta and produce final detections.
    Returns a list of dicts with bbox + 5 landmarks.
    """
    input_w = int(tmeta.network_info.width)
    input_h = int(tmeta.network_info.height)

    # Map name -> layer
    layers_by_name = {}
    for i in range(int(tmeta.num_output_layers)):
        layer = _get_layer(tmeta, i)
        name = _layer_name(layer)
        layers_by_name[name] = layer

    all_dets = []

    for stride in YUNET_STRIDES:
        b = layers_by_name.get(f"bbox_{stride}")
        c = layers_by_name.get(f"cls_{stride}")
        o = layers_by_name.get(f"obj_{stride}")
        k = layers_by_name.get(f"kps_{stride}")
        if not (b and c and o and k):
            # Missing head; skip this stride
            continue

        bbox = _reshape_known(_to_np(b), 4)
        cls = _reshape_known(_to_np(c), 1)
        obj = _reshape_known(_to_np(o), 1)
        kps = _reshape_known(_to_np(k), 10)

        if bbox is None or cls is None or obj is None or kps is None:
            continue

        stride_dets = _decode_yunet_stride_py(
            bbox, cls, obj, kps, stride, input_w, input_h, score_thr
        )
        all_dets.extend(stride_dets)

    final_dets = _nms_py(all_dets, nms_thr)
    return final_dets


# ───────── Display meta helper ─────────
def _acquire_display_meta(batch_meta, frame_meta, cur=None):
    """
    Helper: flush current display meta (if any) to frame and acquire a fresh one.
    """
    if cur is not None:
        pyds.nvds_add_display_meta_to_frame(frame_meta, cur)
    dmeta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    dmeta.num_rects = dmeta.num_lines = dmeta.num_circles = dmeta.num_arrows = (
        dmeta.num_labels
    ) = 0
    return dmeta


# ───────── Probe on PGIE src: decode YuNet from tensor meta + draw / log ─────────
def pgie_src_tensor_probe(pad, info, _):
    """
    For each frame:
      - Find NvDsInferTensorMeta from PGIE (YuNet).
      - Decode bbox + 5 keypoints per face (anchor-free).
      - Run NMS.
      - If NO_DRAW:
          * do NOT add display meta
          * print a compact summary of detections (throttled)
        Else:
          * draw GREEN rectangles for faces
          * draw BIG GREEN circles for the 5 landmarks

    Notes for DeepStream 7.1:
      - NvOSD_CircleParams has: xc, yc, radius, circle_color,
        has_bg_color, bg_color. There is NO line_width field.
      - xc, yc, radius must be ints, so we cast explicitly.
    """
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        bmeta = frame_meta.base_meta.batch_meta

        # Find tensor meta on the frame
        tmeta = None
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if (
                user_meta
                and user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META
            ):
                tmeta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                break
            l_user = l_user.next

        if tmeta is None:
            # No tensor meta on this frame (should not happen if output-tensor-meta=1)
            l_frame = l_frame.next
            continue

        # Decode YuNet heads into final detections
        dets = decode_yunet_from_tensor_meta(tmeta, score_thr=0.75, nms_thr=0.3)
        if not dets:
            if NO_DRAW:
                if frame_meta.frame_num % TENSOR_LOG_EVERY == 0:
                    print(
                        f"[TENSOR] src={frame_meta.source_id} "
                        f"frame={frame_meta.frame_num} dets=0"
                    )
            l_frame = l_frame.next
            continue

        # If --no-draw: just log, don't attach any display meta
        if NO_DRAW:
            # Throttle logging to avoid killing FPS
            if frame_meta.frame_num % TENSOR_LOG_EVERY == 0:
                print(
                    f"[TENSOR] src={frame_meta.source_id} "
                    f"frame={frame_meta.frame_num} dets={len(dets)}"
                )
                # Print first few detections with bbox + score
                for i, d in enumerate(dets[:3]):
                    x1, y1, x2, y2 = d["bbox"]
                    print(
                        f"   det[{i}]: score={d['score']:.3f} "
                        f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                    )
            l_frame = l_frame.next
            continue

        # Drawing mode: attach rects + circles
        fw = int(frame_meta.source_frame_width)
        fh = int(frame_meta.source_frame_height)

        dmeta = None
        max_rects = max_circles = 0

        for det in dets:
            # Acquire (or rollover) display meta
            if dmeta is None:
                dmeta = _acquire_display_meta(bmeta, frame_meta, None)
                max_rects = len(dmeta.rect_params)
                max_circles = len(dmeta.circle_params)

            # Ensure there is room for a new rect
            if dmeta.num_rects >= max_rects:
                dmeta = _acquire_display_meta(bmeta, frame_meta, dmeta)
                max_rects = len(dmeta.rect_params)
                max_circles = len(dmeta.circle_params)

            x1, y1, x2, y2 = det["bbox"]

            # Clamp bbox to frame size (in case model input != frame size)
            x1 = max(0.0, min(x1, fw - 1))
            y1 = max(0.0, min(y1, fh - 1))
            x2 = max(0.0, min(x2, fw - 1))
            y2 = max(0.0, min(y2, fh - 1))

            rect = dmeta.rect_params[dmeta.num_rects]
            rect.left = int(round(x1))
            rect.top = int(round(y1))
            rect.width = int(round(max(0.0, x2 - x1)))
            rect.height = int(round(max(0.0, y2 - y1)))
            rect.border_width = 2
            rect.has_bg_color = 0
            rect.border_color.set(0.0, 1.0, 0.0, 1.0)  # GREEN
            dmeta.num_rects += 1

            # Draw 5 keypoints as big green circles
            for lx, ly in det["landmarks"]:
                if dmeta.num_circles >= max_circles:
                    dmeta = _acquire_display_meta(bmeta, frame_meta, dmeta)
                    max_rects = len(dmeta.rect_params)
                    max_circles = len(dmeta.circle_params)

                # Clamp kpt coords to frame, then cast to int
                lx = max(0.0, min(lx, fw - 1))
                ly = max(0.0, min(ly, fh - 1))

                circ = dmeta.circle_params[dmeta.num_circles]
                circ.xc = int(round(lx))
                circ.yc = int(round(ly))
                circ.radius = int(0.5)
                circ.has_bg_color = 1
                circ.bg_color.set(0.0, 1.0, 0.0, 1.0)
                circ.circle_color.set(0.0, 1.0, 0.0, 1.0)
                dmeta.num_circles += 1

        if dmeta is not None:
            pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# ───────── OSD probe: FPS only ─────────
def osd_sink_probe(pad, info, _):
    """
    Pad probe on nvdsosd sink:
      - Iterate frames in the batch
      - Update FPS counters per source
      - No drawing here (drawing is done in pgie_src_tensor_probe).
    """
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        sid = int(frame_meta.source_id)
        if sid not in perf:
            perf[sid] = FPS(sid)
        perf[sid].update()
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# ───────── uridecodebin handlers (decode + upload to NVMM when needed) ─────────
def on_src_child_added(_parent, obj, name, _u):
    """
    Forward child-added on decodebin, and tune nvv4l2decoder when present.
    """
    if "decodebin" in name:
        obj.connect("child-added", on_src_child_added, None)
    elif "nvv4l2decoder" in name:
        print("[SRC] child-added: nvv4l2decoder – applying HW decode properties")
        for k, v in {
            "drop-frame-interval": 0,
            "num-extra-surfaces": 1,
            "qos": 0,
            "enable-max-performance": 1,
        }.items():
            try:
                obj.set_property(k, v)
            except Exception:
                pass


def on_src_pad_added(decodebin, pad, sinkpad_to_mux):
    """
    When uridecodebin exposes a new pad:
      - If it’s NVMM video, link directly to nvstreammux.
      - Otherwise, insert queue + nvvideoconvert + capsfilter to convert
        into NV12 (NVMM on Jetson, CPU on x86) before nvstreammux.
    """
    caps = pad.get_current_caps() or pad.query_caps()
    if not caps:
        return

    s = caps.get_structure(0)
    media = s.get_name()

    if not media.startswith("video/"):
        print(f"[SRC] pad-added: ignoring non-video pad ({media})")
        return

    feats = caps.get_features(0) if caps.get_size() > 0 else None

    if feats and feats.contains("memory:NVMM"):
        if pad.link(sinkpad_to_mux) != Gst.PadLinkReturn.OK:
            sys.stderr.write("ERROR: link NVMM video -> nvstreammux failed\n")
        else:
            sys.stdout.write(
                f"[SRC] pad-added: {media} (NVMM direct) caps={caps.to_string()}\n"
            )
        return

    pipeline = sinkpad_to_mux.get_parent_element().get_parent()
    q = Gst.ElementFactory.make("queue", None)
    up = force_vic_surface_array(Gst.ElementFactory.make("nvvideoconvert", None))
    capsfilter = Gst.ElementFactory.make("capsfilter", None)

    if IS_JETSON:
        caps_str = f"{NVMM_CAPS},format=NV12"
    else:
        caps_str = "video/x-raw,format=NV12"

    capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))

    for e in (q, up, capsfilter):
        pipeline.add(e)
        e.sync_state_with_parent()

    if pad.link(q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: link src->queue failed\n")
        return
    if not q.link(up) or not up.link(capsfilter):
        sys.stderr.write("ERROR: queue->nvvideoconvert->capsfilter failed\n")
        return

    if capsfilter.get_static_pad("src").link(sinkpad_to_mux) != Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: caps->nvstreammux link failed\n")
        return

    path_note = "(CPU decode → NVMM+VIC)" if IS_JETSON else "(CPU decode → NV12)"
    sys.stdout.write(
        f"[SRC] pad-added: {media} {path_note} "
        f"caps_in={caps.to_string()} caps_out={caps_str}\n"
    )


# ───────── encoder selection (HW first, then SW x264enc) ─────────
def pick_encoder():
    """
    Returns (encoder_name, caps_to_enc_string, parser_name).
    """
    if Gst.ElementFactory.find("nvv4l2h264enc"):
        print("[ENC] Using nvv4l2h264enc (HW H.264)")
        return ("nvv4l2h264enc", f"{NVMM_CAPS},format=NV12", "h264parse")

    if Gst.ElementFactory.find("nvv4l2h265enc"):
        print("[ENC] Using nvv4l2h265enc (HW H.265)")
        return ("nvv4l2h265enc", f"{NVMM_CAPS},format=NV12", "h265parse")

    if Gst.ElementFactory.find("x264enc"):
        print("[ENC] Using x264enc (software)")
        return ("x264enc", "video/x-raw,format=I420", "h264parse")

    raise SystemExit("No encoder found (need nvv4l2h264enc or x264enc)")


# ───────── bus handler ─────────
def on_bus_message(bus, msg, loop):
    mtype = msg.type
    if mtype == Gst.MessageType.EOS:
        print("EOS received")
        loop.quit()
    elif mtype == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        sys.stderr.write(f"ERROR: {err.message}\n{dbg or ''}\n")
        loop.quit()
    elif mtype == Gst.MessageType.WARNING:
        err, dbg = msg.parse_warning()
        sys.stderr.write(f"WARNING: {err.message}\n{dbg or ''}\n")
    return True


# ───────── main pipeline build/run ─────────
def main():
    Gst.init(None)
    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline.new("ds-yunet-faces-mp4")

    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    if not mux:
        raise SystemExit("ERROR: cannot create nvstreammux")

    mux.set_property("batch-size", MUX_BATCH)
    mux.set_property("batched-push-timeout", 25000)
    mux.set_property("width", MUX_W)
    mux.set_property("height", MUX_H)
    mux.set_property("live-source", 0 if SOURCE.startswith("file://") else 1)
    if IS_JETSON:
        try:
            mux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_SURFACE_ARRAY))
        except Exception:
            pass

    pipeline.add(mux)

    src = Gst.ElementFactory.make("uridecodebin", "src")
    if not src:
        raise SystemExit("ERROR: cannot create uridecodebin")
    src.set_property("uri", SOURCE)
    sinkpad = mux.get_request_pad("sink_0")
    src.connect("pad-added", on_src_pad_added, sinkpad)
    src.connect("child-added", on_src_child_added, None)
    pipeline.add(src)

    perf[0] = FPS(0)

    def tick():
        if perf[0].first:
            return True
        cur, avg = perf[0].snapshot()
        sys.stdout.write(f"[PERF] Stream 0 FPS: {cur:.2f} (avg {avg:.2f})\n")
        return True

    GLib.timeout_add(PERF_INTERVAL_SEC * 1000, tick)

    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    if not pgie:
        raise SystemExit("ERROR: cannot create nvinfer (pgie)")
    pgie.set_property("config-file-path", INFER_CONFIG)
    print(f"[PGIE] Using config: {INFER_CONFIG}")

    to_model = force_vic_surface_array(
        Gst.ElementFactory.make("nvvideoconvert", "to_model")
    )
    caps_model = Gst.ElementFactory.make("capsfilter", "caps_model")
    if not to_model or not caps_model:
        raise SystemExit("ERROR: cannot create to_model / caps_model")

    model_caps_str = (
        f"{NVMM_CAPS},format=RGBA" if IS_JETSON else "video/x-raw,format=RGBA"
    )
    caps_model.set_property("caps", Gst.Caps.from_string(model_caps_str))

    to_rgba = force_vic_surface_array(
        Gst.ElementFactory.make("nvvideoconvert", "to_rgba")
    )
    caps_rgba = Gst.ElementFactory.make("capsfilter", "caps_rgba")
    if not to_rgba or not caps_rgba:
        raise SystemExit("ERROR: cannot create to_rgba / caps_rgba")

    rgba_caps_str = (
        f"{NVMM_CAPS},format=RGBA" if IS_JETSON else "video/x-raw,format=RGBA"
    )
    caps_rgba.set_property("caps", Gst.Caps.from_string(rgba_caps_str))

    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    if not osd:
        raise SystemExit("ERROR: cannot create nvdsosd")
    try:
        osd.set_property("process-mode", int(pyds.MODE_GPU))
    except Exception:
        pass

    enc_name, caps_to_enc_str, parser_name = pick_encoder()
    to_enc = force_vic_surface_array(
        Gst.ElementFactory.make("nvvideoconvert", "to_enc")
    )
    caps_to_enc = Gst.ElementFactory.make("capsfilter", "caps_to_enc")
    if not to_enc or not caps_to_enc:
        raise SystemExit("ERROR: cannot create to_enc / caps_to_enc")

    caps_to_enc.set_property("caps", Gst.Caps.from_string(caps_to_enc_str))

    enc = Gst.ElementFactory.make(enc_name, "encoder")
    if not enc:
        raise SystemExit(f"ERROR: cannot create encoder {enc_name}")

    if enc_name.startswith("nvv4l2h26"):
        enc.set_property("bitrate", BITRATE)
        enc.set_property("iframeinterval", GOP)
        try:
            enc.set_property("insert-sps-pps", INSERT_SPS_PPS)
        except Exception:
            pass
        try:
            enc.set_property("maxperf-enable", True)
        except Exception:
            pass
    else:
        enc.set_property("speed-preset", "ultrafast")
        enc.set_property("tune", "zerolatency")
        enc.set_property("bitrate", max(1, BITRATE // 1000))
        enc.set_property("key-int-max", GOP)

    parser = Gst.ElementFactory.make(parser_name, "parser")
    mp4mux = Gst.ElementFactory.make("qtmux", "mp4mux")
    sink = Gst.ElementFactory.make("filesink", "sink")

    if not parser or not mp4mux or not sink:
        raise SystemExit("ERROR: cannot create parser/mp4mux/sink")

    mp4mux.set_property("faststart", True)
    sink.set_property("location", OUT_MP4)
    sink.set_property("sync", False)

    for e in (
        pgie,
        to_model,
        caps_model,
        to_rgba,
        caps_rgba,
        osd,
        to_enc,
        caps_to_enc,
        enc,
        parser,
        mp4mux,
        sink,
    ):
        if not pipeline.add(e):
            raise SystemExit(f"ERROR: failed to add {e.get_name()} to pipeline")

    assert mux.link(to_model)
    assert to_model.link(caps_model)
    assert caps_model.link(pgie)
    assert pgie.link(to_rgba)
    assert to_rgba.link(caps_rgba)
    assert caps_rgba.link(osd)
    assert osd.link(to_enc)
    assert to_enc.link(caps_to_enc)
    assert caps_to_enc.link(enc)
    assert enc.link(parser)
    assert parser.link(mp4mux)
    assert mp4mux.link(sink)

    # Probe after PGIE: decode YuNet from tensor meta + draw/log
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_tensor_probe, None)

    # Probe on OSD sink: FPS only
    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_probe, None)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    print(
        f"\n[PIPELINE] YuNet PGIE: {INFER_CONFIG}\n"
        f"          source: {SOURCE}\n"
        f"          encoder: {enc_name} | caps→encoder: {caps_to_enc_str}\n"
        f"          output MP4: {OUT_MP4}\n"
        f"          streammux: {MUX_W}x{MUX_H}, batch={MUX_BATCH}\n"
        f"          model input caps: {model_caps_str}\n"
        f"          NO_DRAW: {NO_DRAW}\n"
    )

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Ctrl-C received, stopping ...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print(f"[OK] MP4 written → {OUT_MP4}")


# ───────── CLI parsing ─────────
def parse_args():
    global SOURCE, INFER_CONFIG, OUT_MP4
    global MUX_W, MUX_H, MUX_BATCH, BITRATE, GOP, INSERT_SPS_PPS, PERF_INTERVAL_SEC
    global NO_DRAW

    ap = argparse.ArgumentParser(
        "DeepStream YuNet (TensorRT + tensor-meta decode) → MP4 (Jetson + x86)"
    )
    ap.add_argument(
        "-s",
        "--source",
        required=True,
        help="URI: file:///path/video.mp4 or rtsp://... or http(s)://...",
    )
    ap.add_argument(
        "-c",
        "--infer-config",
        required=True,
        help="nvinfer config for YuNet (with engine, output-tensor-meta=1)",
    )
    ap.add_argument(
        "--out",
        default="yunet_out.mp4",
        help="Output MP4 path (default: yunet_out.mp4)",
    )
    ap.add_argument(
        "-w", "--streammux-width", type=int, default=1920, help="nvstreammux width"
    )
    ap.add_argument(
        "-e", "--streammux-height", type=int, default=1080, help="nvstreammux height"
    )
    ap.add_argument(
        "-b",
        "--streammux-batch-size",
        type=int,
        default=1,
        help="nvstreammux batch-size (use 1 for single source)",
    )
    ap.add_argument(
        "--bitrate",
        type=int,
        default=8_000_000,
        help="Target bitrate for encoder (HW: bps, x264: kbps-ish)",
    )
    ap.add_argument("--gop", type=int, default=30, help="GOP / iframeinterval")
    ap.add_argument(
        "--insert-sps-pps",
        action="store_true",
        default=True,
        help="HW enc: insert SPS/PPS for each keyframe",
    )
    ap.add_argument(
        "--perf-interval-sec",
        type=int,
        default=5,
        help="FPS logging interval (seconds)",
    )
    ap.add_argument(
        "--no-draw",
        action="store_true",
        help=(
            "Disable drawing: decode YuNet tensors, run NMS, and print a compact "
            "summary to console (throttled), but do NOT attach any rectangles or "
            "landmarks to the frame. Use this to measure model/pipeline speed "
            "without OSD overlays."
        ),
    )

    a = ap.parse_args()

    if not os.path.isfile(a.infer_config):
        sys.exit(f"ERROR: config not found: {a.infer_config}")

    SOURCE = a.source
    INFER_CONFIG = os.path.abspath(a.infer_config)
    OUT_MP4 = os.path.abspath(a.out)
    out_dir = os.path.dirname(OUT_MP4)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    MUX_W = a.streammux_width
    MUX_H = a.streammux_height
    MUX_BATCH = a.streammux_batch_size
    BITRATE = a.bitrate
    GOP = a.gop
    INSERT_SPS_PPS = a.insert_sps_pps
    PERF_INTERVAL_SEC = a.perf_interval_sec
    NO_DRAW = a.no_draw


if __name__ == "__main__":
    parse_args()
    sys.exit(main())
