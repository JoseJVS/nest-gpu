#include <stdexcept>
#include <string>

#include "coordinate_geometry.h"
#include "mask2d_geometry.h"
#include "test_helpers.h"
#include "tile.h"

using namespace sapi;

void test_square() {
  Tile<Coord2D> square(TILE_SHAPE::RECTANGLE, Coord2D(), {M_SQRT2}, {});
  square.split(1, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {1.1, 0.0}, {0.5}, {});
  auto leaf_tiles = cmask.get_overlapping_leaf_sub_tiles(square);
  assert(!leaf_tiles.empty());
  auto sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 0);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 3);
  assert(leaf_tiles.empty());

  Mask<Coord2D> rmask = construct_mask<Coord2D>(
      MASK_SHAPE::PARALLELOGRAM, {0.0, 0.0}, {0.5, 0.5, -0.5, 0.5}, {});
  leaf_tiles = rmask.get_overlapping_leaf_sub_tiles(square);
  assert(leaf_tiles.size() == 4);
  leaf_tiles.clear();

  Mask<Coord2D> emask = construct_mask<Coord2D>(
      MASK_SHAPE::ELLIPTICAL, {-0.5, 0.0}, {0.2, 0.1, 90}, {});
  leaf_tiles = emask.get_overlapping_leaf_sub_tiles(square);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 1);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 2);
}

void test_triangle() {
  Tile<Coord2D> triangle(TILE_SHAPE::TRIANGLE, construct_coord_2D(0.0, 0.0),
                         {M_SQRT2}, {});
  triangle.split(2, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.0, 0.0}, {0.1}, {});
  assert(cmask.get_overlapping_leaf_sub_tiles(triangle).size() == 4);
  assert(Mask<Coord2D>(cmask).get_overlapping_leaf_sub_tiles(triangle).size() ==
         4);

  Mask<Coord2D> pmask = construct_mask<Coord2D>(
      MASK_SHAPE::PARALLELOGRAM, {1.0, -1.0}, {0.5, 0.5, -0.5, 0.5}, {});
  assert(pmask.get_overlapping_leaf_sub_tiles(triangle).empty());
  assert(Mask<Coord2D>(pmask).get_overlapping_leaf_sub_tiles(triangle).empty());

  Mask<Coord2D> emask = construct_mask<Coord2D>(
      MASK_SHAPE::ELLIPTICAL, {-0.7, -0.8}, {0.2, 0.1, 0}, {});
  assert(emask.get_overlapping_leaf_sub_tiles(triangle).size() == 1);
  assert(Mask<Coord2D>(emask).get_overlapping_leaf_sub_tiles(triangle).size() ==
         1);
}

void test_hexagon() {
  Tile<Coord2D> hexagon(TILE_SHAPE::HEXAGON, construct_coord_2D(0.0, 0.0),
                        {0.1}, {});
  hexagon.split(1, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.0, 0.0}, {1}, {});
  assert(cmask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(cmask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);

  Mask<Coord2D> pmask = construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM,
                                                {-1.0, 1.0}, {-2, 2, 2, 2}, {});
  assert(pmask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(pmask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);

  Mask<Coord2D> emask = construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL,
                                                {0.1, 0.1}, {0.3, 0.2, 0}, {});
  assert(emask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(emask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);
}

void test_updated_dist_check() {
  Mask<Coord2D> pmask = construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM,
                                                {0, 0}, {1, 0, 0, 0.5}, {});
  Mask<Coord2D> emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {1, 0.5}, {});

  Coord2D c;
  c.x_ = 0;
  c.y_ = 1;

  assert(construct_circumscribed_radius<Coord2D>(Coord2D(),
                                                 pmask.bounding_disc_.radius2_)
             .coord_in_radius(c)
             .first &&
         construct_circumscribed_radius<Coord2D>(Coord2D(),
                                                 emask.bounding_disc_.radius2_)
             .coord_in_radius(c)
             .first);

  assert(!pmask.coord_in_mask(c).first && !emask.coord_in_mask(c).first);
}

void test_offset() {
  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1.0}, {});
  Mask<Coord2D> pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});
  Mask<Coord2D> emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {2, 1}, {});

  Coord2D c;
  c.x_ = 2;
  c.y_ = 2;

  assert(!cmask.coord_in_mask(c).first && !pmask.coord_in_mask(c).first &&
         !emask.coord_in_mask(c).first);

  cmask = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1.0}, {2, 1});
  pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {1, 1});
  emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {2, 1}, {0, 2});

  assert(cmask.coord_in_mask(c).first && pmask.coord_in_mask(c).first &&
         emask.coord_in_mask(c).first);
  // The same shapes as blueprints (no origin, so the offset is the anchor)
  // for the two-argument query
  const auto cblueprint =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {1.0}, {2, 1});
  const auto pblueprint =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {1, 1});
  const auto eblueprint =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {}, {2, 1}, {0, 2});
  assert(cblueprint.coord_in_mask(Coord2D(), c).first &&
         pblueprint.coord_in_mask(Coord2D(), c).first &&
         eblueprint.coord_in_mask(Coord2D(), c).first);
}

void test_triangular_mask() {
  Mask<Coord2D> tmask =
      construct_mask<Coord2D>(MASK_SHAPE::TRIANGULAR, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});
  Mask<Coord2D> pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});

  vertidx_t triangular_comparisons = 0;
  for (const auto &x : {M_SQRT2, -M_SQRT2, 1., 0., -1., 0.5, -0.5}) {
    for (const auto &y : {M_SQRT2, -M_SQRT2, 1., 0., -1., 0.5, -0.5}) {
      const Coord2D test_v = construct_coord_2D(x, y);
      if (leq_test(std::fabs(y), std::fabs(x)) && 0 <= x) {
        assert(tmask.coord_in_mask(test_v).first);
        ++triangular_comparisons;
      } else
        assert(!tmask.coord_in_mask(test_v).first);

      assert(pmask.coord_in_mask(test_v).first);
    }
  }

  assert(triangular_comparisons == 16);
}

bool annular_construction_throws(const std::vector<space_t> &params,
                                 const std::string &message) {
  try {
    construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {0, 0}, params, {});
  } catch (const std::invalid_argument &e) {
    return message == e.what();
  }
  return false;
}

void test_annular_mask() {
  // Closed ring 0.5 <= |p - anchor| <= 1 around the anchor (1, 2)
  const Mask<Coord2D> amask =
      construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {1, 2}, {0.5, 1}, {});
  assert(amask.anchor_ == construct_coord_2D(1, 2));
  assert(amask.bounding_disc_.origin_ == construct_coord_2D(1, 2) &&
         almost_equal(amask.bounding_disc_.radius2_, 1) &&
         amask.helper_scalars_.size() == 1 &&
         almost_equal(annulus_inner_radius2(amask), 0.25));

  // Ring points and both boundary circles are in, the hole and the outside out
  for (const auto &c :
       {construct_coord_2D(1.75, 2), construct_coord_2D(1, 1.25),
        construct_coord_2D(1.5, 2), construct_coord_2D(1, 3),
        construct_coord_2D(0, 2), construct_coord_2D(1, 1.5)})
    assert(amask.coord_in_mask(c).first);
  for (const auto &c :
       {construct_coord_2D(1, 2), construct_coord_2D(1.25, 2),
        construct_coord_2D(1.2, 2.2), construct_coord_2D(2.5, 2),
        construct_coord_2D(1, 3.5), construct_coord_2D(2, 3)})
    assert(!amask.coord_in_mask(c).first);

  // Same queries as a blueprint, the offset being the anchor
  const Mask<Coord2D> ablueprint =
      construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {}, {0.5, 1}, {1, 2});
  assert(
      ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(1.75, 2)).first &&
      ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(2, 2)).first &&
      ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(1.5, 2)).first &&
      !ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(1, 2)).first &&
      !ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(1.25, 2)).first &&
      !ablueprint.coord_in_mask(Coord2D(), construct_coord_2D(3, 2)).first);
  // Displacement b - a - anchor
  assert(
      ablueprint
          .coord_in_mask(construct_coord_2D(1, 1), construct_coord_2D(2.75, 3))
          .first &&
      !ablueprint
           .coord_in_mask(construct_coord_2D(1, 1), construct_coord_2D(2, 3))
           .first);

  // r_in = 0 is the circle of radius r_out
  const Mask<Coord2D> dmask =
      construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {0, 0}, {0, 1}, {});
  const Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1}, {});
  for (const auto &x : {-1.5, -1., -0.5, 0., 0.5, 1., 1.5})
    for (const auto &y : {-1.5, -1., -0.5, 0., 0.5, 1., 1.5}) {
      const Coord2D c = construct_coord_2D(x, y);
      assert(dmask.coord_in_mask(c).first == cmask.coord_in_mask(c).first);
    }

  // Invalid parameters
  const std::string radii = "Invalid annular mask radii";
  assert(
      // construct_mask rejects empty params before the shape initializer
      annular_construction_throws({}, "Invalid mask params vector") &&
      annular_construction_throws({1}, "Invalid annular mask params vector") &&
      annular_construction_throws({0, 1, 2},
                                  "Invalid annular mask params vector") &&
      annular_construction_throws({1, 1}, radii) &&
      annular_construction_throws({2, 1}, radii) &&
      annular_construction_throws({-0.5, 1}, radii) &&
      annular_construction_throws({0, 0}, radii) &&
      annular_construction_throws({0, 1e-12}, radii));
}

void test_annular_tiles() {
  const Mask<Coord2D> amask =
      construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {0, 0}, {0.5, 1}, {});

  // Small annulus inside a big tile: the tile holds the centre and meets the
  // ring
  const Mask<Coord2D> small =
      construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {0, 0}, {0.1, 0.2}, {});
  Tile<Coord2D> big(TILE_SHAPE::RECTANGLE, Coord2D(), {10}, {});
  assert(small.tile_overlap(big) == OVERLAP_LEVEL::PARTIAL);

  // Small tile wholly inside the hole
  Tile<Coord2D> in_hole(TILE_SHAPE::RECTANGLE, Coord2D(), {0.1}, {});
  assert(amask.tile_overlap(in_hole) == OVERLAP_LEVEL::NONE);
  assert(amask.get_overlapping_leaf_sub_tiles(in_hole).empty());

  // Small tile wholly inside the ring, and wholly outside
  Tile<Coord2D> in_ring(TILE_SHAPE::RECTANGLE, construct_coord_2D(0.75, 0),
                        {0.1}, {});
  assert(amask.tile_overlap(in_ring) == OVERLAP_LEVEL::FULL);
  Tile<Coord2D> outside(TILE_SHAPE::RECTANGLE, construct_coord_2D(3, 0), {0.1},
                        {});
  assert(amask.tile_overlap(outside) == OVERLAP_LEVEL::NONE);
}

int main() {
  test_square();
  test_triangle();
  test_hexagon();
  test_updated_dist_check();
  test_offset();
  test_triangular_mask();
  test_annular_mask();
  test_annular_tiles();
  return 0;
}
