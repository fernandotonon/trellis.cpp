#include "preprocess.h"

#include <cstdio>
#include <stdexcept>

// Expected windows come from upstream TRELLIS.2 preprocess_image + PIL crop
// (center -/+ size//2, PIL rounds half to even).
static void check(int xmin, int ymin, int xmax, int ymax, float m, int ex0, int ey0, int esz) {
    int x0, y0, sz;
    trellis::crop_window(xmin, ymin, xmax, ymax, m, x0, y0, sz);
    if (x0 != ex0 || y0 != ey0 || sz != esz) {
        fprintf(stderr, "crop (%d,%d,%d,%d) x%.2f: got %d,%d,%d want %d,%d,%d\n",
                xmin, ymin, xmax, ymax, m, x0, y0, sz, ex0, ey0, esz);
        throw std::runtime_error("crop window test failed");
    }
}

int main() {
    check(50, 100, 950, 700, 1.0f, 50, -50, 900);      // bbox 50..950 -> 900 px, not 902
    check(49, 0, 950, 10, 1.0f, 50, -445, 900);        // center 499.5: 49.5 -> 50 (even)
    check(51, 0, 952, 10, 1.0f, 52, -445, 900);        // center 501.5: 51.5 -> 52 (even)
    check(10, 20, 101, 60, 1.0f, 10, -5, 90);          // odd size 91 -> 90, center 55.5: 10.5 -> 10
    check(0, 0, 99, 99, 1.1f, -4, -4, 108);            // size int(99*1.1)=108, -4.5 -> -4
    check(50, 100, 950, 700, 0.0f, 4, -96, 992);       // legacy: half int(451*1.10)=496
    printf("crop window tests passed\n");
    return 0;
}
