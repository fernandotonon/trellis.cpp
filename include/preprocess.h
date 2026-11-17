// Image preprocessing for TRELLIS.2: load -> white-bg removal -> square crop ->
// premultiply on black -> resize -> ImageNet normalize -> [3,S,S] torch CHW.
#pragma once
#include <vector>
#include <string>

namespace trellis {
struct Model;
// Returns [3*S*S] float in torch CHW order (== ggml [S,S,3,1]), or empty on failure.
std::vector<float> preprocess_image(const std::string& path, int S = 512);

// Non-neural background removal: existing alpha if present, else white-bg threshold;
// bbox-crop + premultiply on transparent black -> square RGBA uint8 cutout (size in `sz`).
// margin <= 0 keeps this port's legacy crop (side = 2*int((bbox/2 + 1) * 1.10));
// margin > 0 reproduces upstream TRELLIS.2 preprocess_image exactly for that factor
// (side = int(bbox * margin), PIL crop at center -/+ side//2 rounded half-to-even).
std::vector<unsigned char> threshold_cutout(const std::string& path, int& sz, float margin = 0.0f);

// Square crop window for a subject bbox (inclusive pixel bounds), as described above:
// left/top corner in x0/y0, side in sz. Exposed for tests.
void crop_window(int xmin, int ymin, int xmax, int ymax, float margin, int& x0, int& y0, int& sz);

// True when the image carries a real (not all-opaque) alpha channel, i.e. it is
// pre-matted and needs no background removal.
bool image_has_alpha(const std::string& path);

// BiRefNet background removal: run the matte, restore it to the source aspect ratio, then
// bbox-crop and premultiply -> square RGBA uint8 cutout
// (size returned in `sz`). bm = loaded birefnet model, gpu = device for the deform kernel.
std::vector<unsigned char> birefnet_cutout(const std::string& path, const Model& bm, int gpu, int& sz,
                                           float margin = 0.0f);
// Resize a square RGB/RGBA-uint8 cutout to SxS, ImageNet-normalize -> [3,S,S] torch CHW.
std::vector<float> normalize_cutout(const std::vector<unsigned char>& rgb, int sz, int S);

// Resize a square RGB-uint8 cutout to SxS and scale to [0,1] WITHOUT ImageNet normalization
// -> [3,S,S] torch CHW. The NAF guide branch (Pixal3D) consumes the raw [0,1] image, not the
// normalized one that feeds DINOv3.
std::vector<float> cutout_to_chw01(const std::vector<unsigned char>& rgb, int sz, int S);
} // namespace trellis
