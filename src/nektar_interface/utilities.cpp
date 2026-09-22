#include <nektar_interface/utilities.hpp>

namespace NESO {

std::mt19937 uniform_within_elements(
    Nektar::SpatialDomains::MeshGraphSharedPtr graph, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    const REAL tol, std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  std::map<int, Nektar::SpatialDomains::Geometry2D *> geoms_2d;
  std::map<int, Nektar::SpatialDomains::Geometry3D *> geoms_3d;
  int npart_total;
  int nelements;

  if (ndim == 2) {
    get_all_elements_2d(graph, geoms_2d);
    nelements = geoms_2d.size();
  } else if (ndim == 3) {
    get_all_elements_3d(graph, geoms_3d);
    nelements = geoms_3d.size();
  }
  npart_total = nelements * npart_per_cell;

  positions.resize(ndim);
  cells.resize(npart_total);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    Array<OneD, NekDouble> local_coord(3);
    auto bb = geom->GetBoundingBox();
    coord[0] = 0.0;
    coord[1] = 0.0;
    coord[2] = 0.0;

    auto lambda_sample_new = [&]() {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL bound_lower = bb[dx];
        const REAL bound_upper = bb[dx + 3];
        std::uniform_real_distribution<double> dist(bound_lower, bound_upper);
        coord[dx] = dist(rng);
      }
    };

    lambda_sample_new();
    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_dispatch = [&](auto container) {
    Array<OneD, NekDouble> coord(3);
    int ex = 0;
    int index = 0;
    for (auto id_element : container) {
      for (int px = 0; px < npart_per_cell; px++) {
        lambda_sample(id_element.second, coord);
        for (int dx = 0; dx < ndim; dx++) {
          positions.at(dx).at(index) = coord[dx];
        }
        cells.at(index) = ex;
        index++;
      }
      ex++;
    }
  };

  if (ndim == 2) {
    lambda_dispatch(geoms_2d);
  } else if (ndim == 3) {
    lambda_dispatch(geoms_3d);
  }

  return rng;
}

extern "C" {
void F77NAME(dgelsd)(
    const int &m,        // Number of rows in A
    const int &n,        // Number of columns in A
    const int &nrhs,     // Number of right-hand sides
    double *a,           // Matrix A
    const int &lda,      // Leading dimension of A
    double *b,           // Matrix B
    const int &ldb,      // Leading dimension of B
    double *s,           // Output singular values
    const double *rcond, // Cutoff value for singular values
    int &rank,           // Output effective rank
    double *work,        // Workspace array pointer
    const int &lwork,    // Size of workspace array (-1 to query optimal size)
    int *iwork,          // Integer workspace array pointer
    int &info            // Output status (0 = success)
);
}

std::mt19937
weighted_within_elements(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                         ExpListSharedPtr exp_list, const int npart_per_cell,
                         std::vector<std::vector<double>> &positions,
                         std::vector<int> &cells, std::vector<double> &weights,
                         const REAL tol, std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  std::map<int, Nektar::SpatialDomains::Geometry2D *> geoms_2d;
  std::map<int, Nektar::SpatialDomains::Geometry3D *> geoms_3d;
  int npart_total;
  int nelements;

  if (ndim == 2) {
    get_all_elements_2d(graph, geoms_2d);
    nelements = geoms_2d.size();
  } else if (ndim == 3) {
    get_all_elements_3d(graph, geoms_3d);
    nelements = geoms_3d.size();
  }
  npart_total = nelements * npart_per_cell;

  positions.resize(ndim);
  cells.resize(npart_total);
  weights.resize(npart_total);

  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    Array<OneD, NekDouble> local_coord(3);
    auto bb = geom->GetBoundingBox();
    coord[0] = 0.0;
    coord[1] = 0.0;
    coord[2] = 0.0;

    auto lambda_sample_new = [&]() {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL bound_lower = bb[dx];
        const REAL bound_upper = bb[dx + 3];
        std::uniform_real_distribution<double> dist(bound_lower, bound_upper);
        coord[dx] = dist(rng);
      }
    };

    lambda_sample_new();
    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_all_weights_positive = [&](int index) {
    bool all_positive = true;
    for (int i = index; i < index + npart_per_cell; ++i) {
      if (weights.at(i) < 0) {
        all_positive = false;
        break;
      }
    }
    return all_positive;
  };

  auto lambda_volume = [&](auto geom) {
    LibUtilities::ShapeType shape = geom->GetShapeType();
    switch (shape) {
    case LibUtilities::ShapeType::eQuadrilateral:
      return 4.0;
    case LibUtilities::ShapeType::eTriangle:
      return 2.0;
    case LibUtilities::ShapeType::eTetrahedron:
      return 2.0 / 3.0;
    case LibUtilities::ShapeType::ePyramid:
      return 4.0 / 3.0;
    case LibUtilities::ShapeType::ePrism:
      return 4.0;
    case LibUtilities::ShapeType::eHexahedron:
      return 8.0;
    default:
      return 1.0;
    }
  };

  auto lambda_dispatch = [&](auto container) {
    Array<OneD, NekDouble> coord(3);
    Array<OneD, NekDouble> loccoord(3);
    int index = 0;
    int ex = 0;
    for (auto &[g_id, geom] : container) {

      auto exp = exp_list->GetExp(exp_list->GetElmtToExpId(g_id));
      size_t ncoeffs = exp->GetNcoeffs();

      StdRegions::StdMatrixKey mkey(StdRegions::MatrixType::eMass,
                                    exp->DetShapeType(), *exp);

      auto Mass = exp->CreateGeneralMatrix(mkey);

      int M = ncoeffs * (ncoeffs + 1) / 2;
      Array<OneD, NekDouble> Mv(M);
      int tri = 0;
      for (int m1 = 0; m1 < ncoeffs; ++m1) {
        for (int m2 = m1; m2 < ncoeffs; ++m2) {
          Mv[tri++] = Mass->GetValue(m1, m2);
        }
      }

      auto jac = exp->GetGeomFactors()->GetJac();
      double vol = lambda_volume(geom);
      NekVector<double> NW(npart_per_cell,
                           Array<OneD, NekDouble>(
                               npart_per_cell, vol * jac[0] / npart_per_cell));

      Array<OneD, NekDouble> S(ncoeffs);
      Array<OneD, NekDouble> SST(M * npart_per_cell);
      bool all_positive = true;

      do {
        for (int px = 0; px < npart_per_cell; px++) {
          lambda_sample(exp->GetGeom(), coord);
          for (int dx = 0; dx < ndim; dx++) {
            positions.at(dx).at(index + px) = coord[dx];
          }
          cells.at(index + px) = ex;
          exp->GetGeom()->GetLocCoords(coord, loccoord);

          for (int m = 0; m < ncoeffs; ++m) {
            S[m] = exp->PhysEvaluateBasis(loccoord, m);
          }

          int tri = 0;
          for (int m1 = 0; m1 < ncoeffs; ++m1) {
            for (int m2 = m1; m2 < ncoeffs; ++m2) {
              SST[px * M + tri++] = S[m1] * S[m2];
            }
          }
        }
        NekMatrix<double> Kmat(M, npart_per_cell, SST);

        NekVector<double> Mvec(M, Mv);
        NekVector<double> Bvec = Mvec - Kmat * NW;

        int N = Kmat.GetColumns();
        double rcond = -1.0;
        double wkopt;
        double *work;
        int info, lwork, rank;

        int minmn = std::min(M, N);
        int maxmn = std::max(M, N);
        int smlsiz = 25;
        int nlvl = std::max(0, int(std::log2(minmn / (smlsiz + 1))) + 1);

        int *iwork = (int *)malloc((minmn * (3 * nlvl + 11) * sizeof(int)));
        lwork = -1;
        double *s = (double *)malloc(minmn * sizeof(double));

        Array<OneD, NekDouble> Dv(std::max(M, npart_per_cell), 0.0);
        Vmath::Vcopy(M, Bvec.GetPtr(), 1, Dv, 1);

        F77NAME(dgelsd)(M, N, 1, Kmat.GetRawPtr(), M, Dv.data(), maxmn, s,
                        &rcond, rank, &wkopt, lwork, iwork, info);
        lwork = (int)wkopt;
        work = (double *)malloc(lwork * sizeof(double));
        F77NAME(dgelsd)(M, N, 1, Kmat.GetRawPtr(), M, Dv.data(), maxmn, s,
                        &rcond, rank, work, lwork, iwork, info);
        free((void *)work);
        free((void *)s);
        free((void *)iwork);

        NekVector<double> Dvec(npart_per_cell, Dv);

        all_positive = true;
        for (int px = 0; px < npart_per_cell; px++) {
          double weight = NW.GetPtr()[px] + Dvec.GetPtr()[px];
          if (weight < 0) {
            all_positive = false;
            break;
          }
          weights.at(index + px) = weight;
        }
      } while (!all_positive);

      index += npart_per_cell;
      ex++;
    }
  };

  if (ndim == 2) {
    lambda_dispatch(geoms_2d);
  } else if (ndim == 3) {
    lambda_dispatch(geoms_3d);
  }

  return rng;
}

std::mt19937
uniform_within_composite(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                         const int compid, const int npart_per_cell,
                         std::vector<std::vector<double>> &positions,
                         std::vector<int> &cells, const REAL tol,
                         std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  auto geoms = graph->GetComposite(compid)->m_geomVec;

  int npart_total;
  int nelements;

  npart_total = nelements * npart_per_cell;

  positions.resize(ndim);
  cells.resize(npart_total);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    Array<OneD, NekDouble> local_coord(3);
    auto bb = geom->GetBoundingBox();
    coord[0] = 0.0;
    coord[1] = 0.0;
    coord[2] = 0.0;

    auto lambda_sample_new = [&]() {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL bound_lower = bb[dx];
        const REAL bound_upper = bb[dx + 3];
        std::uniform_real_distribution<double> dist(bound_lower, bound_upper);
        coord[dx] = dist(rng);
      }
    };

    lambda_sample_new();
    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_dispatch = [&](auto container) {
    Array<OneD, NekDouble> coord(3);
    int ex = 0;
    int index = 0;
    for (auto element : container) {
      for (int px = 0; px < npart_per_cell; px++) {
        lambda_sample(element, coord);
        for (int dx = 0; dx < ndim; dx++) {
          positions.at(dx).at(index) = coord[dx];
        }
        cells.at(index) = ex;
        index++;
      }
      ex++;
    }
  };

  lambda_dispatch(geoms);

  return rng;
}

std::mt19937
dist_within_extents(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                    Nektar::LibUtilities::EquationSharedPtr eqn, const double t,
                    const int npart,
                    std::vector<std::vector<double>> &positions,
                    std::vector<int> &cells, const REAL tol,
                    std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  std::map<int, Nektar::SpatialDomains::Geometry2D *> geoms_2d;
  std::map<int, Nektar::SpatialDomains::Geometry3D *> geoms_3d;
  int nelements;

  if (ndim == 2) {
    get_all_elements_2d(graph, geoms_2d);
    nelements = geoms_2d.size();
  } else if (ndim == 3) {
    get_all_elements_3d(graph, geoms_3d);
    nelements = geoms_3d.size();
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    Array<OneD, NekDouble> local_coord(3);
    auto bb = geom->GetBoundingBox();
    coord[0] = 0.0;
    coord[1] = 0.0;
    coord[2] = 0.0;

    auto lambda_sample_new = [&]() {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL bound_lower = bb[dx];
        const REAL bound_upper = bb[dx + 3];
        std::uniform_real_distribution<double> dist(bound_lower, bound_upper);
        coord[dx] = dist(rng);
      }
    };

    lambda_sample_new();

    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_dispatch = [&](auto &container) {
    Array<OneD, NekDouble> coord(3);
    std::vector<double> weight_per_cell(nelements, 0);
    std::vector<int> flat_idx(nelements);
    double local_weight = 0;
    auto lambda_preprocess = [&](auto &geoms) {
      double x, y, z;
      int ex = 0;
      for (const auto &[id, geom] : geoms) {
        int Nv = geom->GetNumVerts();
        for (int v = 0; v < Nv; ++v) {
          geom->GetVertex(v)->GetCoords(x, y, z);
          double P = eqn->Evaluate(x, y, z, t);
          NESOASSERT(P >= 0 && P <= 1, "Probability distribution must be "
                                       "between 0 and 1, but evaluates to " +
                                           std::to_string(P));
          weight_per_cell[ex] += P / Nv;
        }
        local_weight += weight_per_cell[ex];
        flat_idx[ex] = id;
        ex++;
      }
    };

    lambda_preprocess(container);

    double global_weight = 0;
    MPI_Allreduce(&local_weight, &global_weight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    int npart_local = std::round(npart * local_weight / global_weight);

    if (npart_local > 0) {
      positions.resize(ndim);
      cells.resize(npart_local);
      for (int dimx = 0; dimx < ndim; dimx++) {
        positions[dimx] = std::vector<double>(npart_local);
      }
      std::discrete_distribution disc(weight_per_cell.begin(),
                                      weight_per_cell.end());

      int index = 0;
      while (index < npart_local) {
        auto cell = disc(rng);
        lambda_sample(container[flat_idx[cell]], coord);
        double P = eqn->Evaluate(coord[0], coord[1], coord[2], t);
        NESOASSERT(P >= 0 && P <= 1, "Probability distribution must be "
                                     "between 0 and 1, but evaluates to " +
                                         std::to_string(P));
        std::bernoulli_distribution bern(P);

        if (bern(rng)) {
          for (int dx = 0; dx < ndim; dx++) {
            positions.at(dx).at(index) = coord[dx];
          }
          cells.at(index) = cell;
          index++;
        }
      }
    }
  };

  if (ndim == 2) {
    lambda_dispatch(geoms_2d);
  } else if (ndim == 3) {
    lambda_dispatch(geoms_3d);
  }

  return rng;
}

} // namespace NESO
