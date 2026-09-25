#pragma once

#include "quad_refinement.h"

#include <cstddef>
#include <string>
#include <vector>

namespace trellis {

void write_quad_refinement_artifacts(
    const std::string& output_path,
    const QuadRefinementResult& result,
    const std::vector<size_t>& parent_face_sizes);

}
