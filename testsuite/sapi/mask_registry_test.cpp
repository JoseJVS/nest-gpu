#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

#include "mask2d_geometry.h"
#include "mask_creators.h"
#include "tile.h"

using namespace sapi;

int main() {
  CreatorRegistry<Mask<Coord2D>> mkr;
  initialize_mk_registry(mkr);

  auto m = mkr.get_creator("circular")->create({}, {0.5}, {});
  assert(m.shape_ == MASK_SHAPE::CIRCULAR &&
         almost_equal(m.bounding_disk_.radius2_, 0.25) && m.anchor_.is_null() &&
         m.helper_scalars_.empty() && m.helper_vectors_.empty());

  m = mkr.get_creator("parallelogram")->create({1, 2}, {1, 1, -1, 1}, {});
  assert(m.shape_ == MASK_SHAPE::PARALLELOGRAM &&
         almost_equal(m.bounding_disk_.radius2_, 2) &&
         m.anchor_ == construct_coord_2D(1, 2) &&
         m.bounding_disk_.origin_ == construct_coord_2D(1, 2) &&
         m.helper_scalars_.size() == 1 &&
         almost_equal(m.helper_scalars_.at(0), 4) &&
         m.helper_vectors_.size() == 4 &&
         m.helper_vectors_.at(3) == construct_coord_2D(-1, 1));

  // The ellipse stores the (cos, sin) of its rotation
  m = mkr.get_creator("elliptical")->create({}, {1, 0.5, 90}, {});
  assert(m.shape_ == MASK_SHAPE::ELLIPTICAL &&
         almost_equal(m.bounding_disk_.radius2_, 1) && m.anchor_.is_null() &&
         m.helper_scalars_.empty() && m.helper_vectors_.size() == 2 &&
         m.helper_vectors_.at(0) == construct_coord_2D(1, 0.25) &&
         m.helper_vectors_.at(1) == construct_coord_2D(0, 1));

  m = mkr.get_creator("triangular")->create({}, {1, 1, -1, 1}, {10, 1});
  // Smallest enclosing disk of the right triangle {0, (1,1), (-1,1)} anchored
  // at the offset (10, 1): centered on the hypotenuse midpoint, radius^2 1
  // widened by 2 EXACT_FILTER_REL_TOL
  assert(m.shape_ == MASK_SHAPE::TRIANGULAR &&
         leq_test(1, m.bounding_disk_.radius2_) &&
         m.bounding_disk_.radius2_ <= 1 + 2 * EXACT_FILTER_REL_TOL &&
         m.anchor_ == construct_coord_2D(10, 1) &&
         m.bounding_disk_.origin_ == construct_coord_2D(10, 2) &&
         m.helper_scalars_.size() == 1 &&
         almost_equal(m.helper_scalars_.at(0), 2) &&
         m.helper_vectors_.size() == 2);

  m = mkr.get_creator("annular")->create({1, 2}, {0.5, 2}, {});
  assert(m.shape_ == MASK_SHAPE::ANNULAR &&
         almost_equal(m.bounding_disk_.radius2_, 4) &&
         m.anchor_ == construct_coord_2D(1, 2) &&
         m.bounding_disk_.origin_ == construct_coord_2D(1, 2) &&
         m.helper_scalars_.size() == 1 &&
         almost_equal(m.helper_scalars_.at(0), 0.25) &&
         m.helper_vectors_.empty());

  return 0;
}
