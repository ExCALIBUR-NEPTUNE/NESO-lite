#include <nektar_interface/geometry_transport/geometry_transport_2d.hpp>

namespace NESO {

/**
 * Get all 2D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry2D pointers.
 */
void get_all_elements_2d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, Nektar::SpatialDomains::Geometry2D *> &geoms) {
  geoms.clear();

  for (const auto &e : graph->GetGeomMap<SpatialDomains::TriGeom>()) {
    geoms[e.first] = e.second;
  }
  for (const auto &e : graph->GetGeomMap<SpatialDomains::QuadGeom>()) {
    geoms[e.first] = e.second;
  }
}

/**
 * Get a local 2D geometry object from a Nektar++ MeshGraph
 *
 * @param graph Nektar++ MeshGraph to return geometry object from.
 * @returns Local 2D geometry object.
 */
Geometry2D *get_element_2d(Nektar::SpatialDomains::MeshGraphSharedPtr &graph) {
  {
    auto geoms = graph->GetGeomMap<SpatialDomains::QuadGeom>();
    if (geoms.size() > 0) {
      return (*geoms.begin()).second;
    }
  }
  auto geoms = graph->GetGeomMap<SpatialDomains::TriGeom>();
  NESOASSERT(geoms.size() > 0, "No local 2D geometry objects found.");
  return (*geoms.begin()).second;
}

} // namespace NESO
