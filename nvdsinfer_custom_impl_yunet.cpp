/*REFRENCE FILE FOR PARSING THE DATA IF KPTS NOT RELAVENT*/

#include <nvdsinfer_custom_impl.h>
#include <glib.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

    /*
     * YuNet custom parser for DeepStream 7.1 (anchor-free decoder)
     *
     * NOTE ABOUT CURRENT PIPELINE:
     * ----------------------------
     * In the updated pipeline we set:
     *
     *     output-tensor-meta = 1
     *
     * in the nvinfer config, and we decode YuNet (bbox + 5 keypoints) on the
     * Python side from NvDsInferTensorMeta. That means:
     *
     *   • This C++ parser is **no longer required** for the YuNet → MP4 script.
     *   • You can:
     *       - Remove `parse-bbox-func-name` and `custom-lib-path` from the
     *         YuNet nvinfer config, OR
     *       - Keep them for other apps that still want C++ decoding.
     *
     * The code below is left intact as a reference implementation of the
     * anchor-free decode (same math as we now use in Python).
     */

    namespace {

// Number of classes (face vs background). YuNet is single-class.
static const int kNumClasses = 1;

// Stride per detection head. For each stride we have a dense grid of
// (inputW/stride) × (inputH/stride) cells, and ONE “anchor” (grid cell)
// per location → anchor-free formulation.
static const std::vector<int> kStrides = {8, 16, 32};

// Default thresholds; can be overridden in DeepStream config via
// perClassPreclusterThreshold / perClassPostclusterThreshold.
static const float kDefaultScoreThreshold = 0.6f;
static const float kDefaultNmsThreshold   = 0.3f;

/**
 * Detection
 * ---------
 * Internal representation of a YuNet detection:
 *  • (x1,y1,x2,y2) in pixel coordinates w.r.t network input size.
 *  • landmarks: 5 points × (x,y) = 10 floats (also in pixel coords).
 *  • score: final confidence for this face (combined cls × obj).
 */
struct Detection {
    float x1, y1, x2, y2;
    float landmarks[10];  // 5 (x,y) pairs
    float score;
};

inline int getDimsSize(const NvDsInferDims &d) {
    int s = 1;
    for (int i = 0; i < d.numDims; ++i) s *= d.d[i];
    return s;
}

float IoU(const Detection &a, const Detection &b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float w  = std::max(0.0f, x2 - x1);
    float h  = std::max(0.0f, y2 - y1);
    float inter = w * h;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni   = areaA + areaB - inter;

    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

void nms(const std::vector<Detection> &in, float nmsThresh,
         std::vector<Detection> &out) {
    out.clear();
    if (in.empty()) return;

    std::vector<int> idxs(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        idxs[i] = static_cast<int>(i);

    std::sort(idxs.begin(), idxs.end(),
              [&](int a, int b) { return in[a].score > in[b].score; });

    std::vector<bool> suppressed(in.size(), false);

    for (size_t _i = 0; _i < idxs.size(); ++_i) {
        int i = idxs[_i];
        if (suppressed[i]) continue;

        out.push_back(in[i]);

        for (size_t _j = _i + 1; _j < idxs.size(); ++_j) {
            int j = idxs[_j];
            if (suppressed[j]) continue;

            if (IoU(in[i], in[j]) > nmsThresh) {
                suppressed[j] = true;
            }
        }
    }
}

const NvDsInferLayerInfo* findLayer(const std::vector<NvDsInferLayerInfo> &layers,
                                    const char* name) {
    for (const auto &l : layers) {
        if (!std::strcmp(l.layerName, name)) {
            return &l;
        }
    }
    return nullptr;
}

void decodeYuNetStride(const float *bbox,
                       const float *cls,
                       const float *obj,
                       const float *kps,
                       int numAnchors,
                       int inputW,
                       int inputH,
                       float scoreThresh,
                       std::vector<Detection> &dets,
                       int strideIdx)
{
    if (!bbox || !cls || !obj || !kps || numAnchors <= 0) {
        return;
    }

    int stride = kStrides[strideIdx];
    int cols   = inputW / stride;
    int rows   = inputH / stride;

    if (cols * rows != numAnchors) {
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;

            float cls_score = cls[idx];
            float obj_score = obj[idx];

            cls_score = std::min(std::max(cls_score, 0.0f), 1.0f);
            obj_score = std::min(std::max(obj_score, 0.0f), 1.0f);

            float score = std::sqrt(cls_score * obj_score);
            if (score < scoreThresh)
                continue;

            float dx = bbox[idx * 4 + 0];
            float dy = bbox[idx * 4 + 1];
            float dw = bbox[idx * 4 + 2];
            float dh = bbox[idx * 4 + 3];

            float cx = (static_cast<float>(c) + dx) * stride;
            float cy = (static_cast<float>(r) + dy) * stride;
            float w  = std::exp(dw) * stride;
            float h  = std::exp(dh) * stride;

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.score = score;

            for (int n = 0; n < 5; ++n) {
                float lx_off = kps[idx * 10 + 2 * n + 0];
                float ly_off = kps[idx * 10 + 2 * n + 1];

                float lx = (lx_off + static_cast<float>(c)) * stride;
                float ly = (ly_off + static_cast<float>(r)) * stride;

                det.landmarks[2 * n + 0] = lx;
                det.landmarks[2 * n + 1] = ly;
            }

            det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(inputW  - 1)));
            det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(inputH - 1)));
            det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(inputW  - 1)));
            det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(inputH - 1)));

            dets.push_back(det);
        }
    }
}

} // namespace

extern "C" bool
NvDsInferParseCustomYUNet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    // NOTE:
    //   For the Python YuNet → MP4 script we now decode from tensor meta instead of
    //   using this parser. You can safely disable this function in the config.
    //
    //   If you still use this parser in some other app, the implementation below
    //   keeps the original behavior (bbox-only, no landmarks exported to DeepStream).

    objectList.clear();

    const char* bboxNames[3] = {"bbox_8", "bbox_16", "bbox_32"};
    const char* clsNames[3]  = {"cls_8",  "cls_16",  "cls_32"};
    const char* objNames[3]  = {"obj_8",  "obj_16",  "obj_32"};
    const char* kpsNames[3]  = {"kps_8",  "kps_16",  "kps_32"};

    const NvDsInferLayerInfo* bboxLayers[3] = {nullptr, nullptr, nullptr};
    const NvDsInferLayerInfo* clsLayers[3]  = {nullptr, nullptr, nullptr};
    const NvDsInferLayerInfo* objLayers[3]  = {nullptr, nullptr, nullptr};
    const NvDsInferLayerInfo* kpsLayers[3]  = {nullptr, nullptr, nullptr};

    for (int i = 0; i < 3; ++i) {
        bboxLayers[i] = findLayer(outputLayersInfo, bboxNames[i]);
        clsLayers[i]  = findLayer(outputLayersInfo, clsNames[i]);
        objLayers[i]  = findLayer(outputLayersInfo, objNames[i]);
        kpsLayers[i]  = findLayer(outputLayersInfo, kpsNames[i]);
        if (!bboxLayers[i] || !clsLayers[i] || !objLayers[i] || !kpsLayers[i]) {
            return false;
        }
    }

    int numPriorsTotal = 0;
    int numAnchors[3]  = {0, 0, 0};

    for (int i = 0; i < 3; ++i) {
        int bboxSize = getDimsSize(bboxLayers[i]->inferDims);
        int clsSize  = getDimsSize(clsLayers[i]->inferDims);
        int objSize  = getDimsSize(objLayers[i]->inferDims);
        int kpsSize  = getDimsSize(kpsLayers[i]->inferDims);

        if (bboxSize % 4 != 0 || kpsSize % 10 != 0) {
            return false;
        }

        numAnchors[i] = bboxSize / 4;

        if (clsSize != numAnchors[i] || objSize != numAnchors[i] ||
            kpsSize != numAnchors[i] * 10) {
            return false;
        }

        numPriorsTotal += numAnchors[i];
    }

    float scoreThresh = kDefaultScoreThreshold;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        scoreThresh = detectionParams.perClassPreclusterThreshold[0];
    }

    float nmsThresh = kDefaultNmsThreshold;
    if (!detectionParams.perClassPostclusterThreshold.empty()) {
        nmsThresh = detectionParams.perClassPostclusterThreshold[0];
    }

    std::vector<Detection> dets;
    dets.reserve(numPriorsTotal);

    int inputW = static_cast<int>(networkInfo.width);
    int inputH = static_cast<int>(networkInfo.height);

    for (int i = 0; i < 3; ++i) {
        const float* bbox = static_cast<const float*>(bboxLayers[i]->buffer);
        const float* cls  = static_cast<const float*>(clsLayers[i]->buffer);
        const float* obj  = static_cast<const float*>(objLayers[i]->buffer);
        const float* kps  = static_cast<const float*>(kpsLayers[i]->buffer);

        decodeYuNetStride(
            bbox, cls, obj, kps,
            numAnchors[i],
            inputW,
            inputH,
            scoreThresh,
            dets,
            i
        );
    }

    std::vector<Detection> finalDets;
    nms(dets, nmsThresh, finalDets);

    for (const auto &d : finalDets) {
        NvDsInferParseObjectInfo o;
        o.left   = d.x1;
        o.top    = d.y1;
        o.width  = d.x2 - d.x1;
        o.height = d.y2 - d.y1;
        o.detectionConfidence = d.score;
        o.classId = 0;  // single-class: face
        objectList.push_back(o);
    }

    return true;
}

extern "C" bool NvDsInferInitialize(NvDsInferContextInitParams const *) {
    return true;
}

extern "C" void NvDsInferDeInitialize(void) {
}
