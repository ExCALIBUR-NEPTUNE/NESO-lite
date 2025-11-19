#include "../../../include/nektar_interface/solver_base/partsys_base.hpp"
#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <type_traits>

namespace NESO::Particles {

ParticleSystemFactory &GetParticleSystemFactory() {
  static ParticleSystemFactory instance;
  return instance;
}

PartSysBase::PartSysBase(const NESOReaderSharedPtr config,
                         const SD::MeshGraphSharedPtr graph, MPI_Comm comm,
                         PartSysOptions options)
    : config(config), graph(graph), comm(comm),
      ndim(graph->GetSpaceDimension()) {

  // Store options
  this->options = options;

  // Create interface between particles and nektar++
  this->particle_mesh_interface =
      std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
  extend_halos_fixed_offset(this->options.extend_halos_offset,
                            this->particle_mesh_interface);
  this->sycl_target =
      std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
  this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
      this->sycl_target, this->particle_mesh_interface);
  this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                          this->nektar_graph_local_mapper);
}

/**
 * @details For each entry in the param_vals map (constructed via report_param),
 * write the value to stdout
 * @see also report_param()
 */
void PartSysBase::add_params_report() {

  std::cout << "Particle settings:" << std::endl;
  for (auto const &[param_lbl, param_str_val] : this->param_vals_to_report) {
    std::cout << "  " << param_lbl << ": " << param_str_val << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl
            << std::endl;
}

void PartSysBase::free() {
  if (this->h5part) {
    this->h5part->close();
  }
  this->particle_group->free();
  this->sycl_target->free();
  this->particle_mesh_interface->free();
}

bool PartSysBase::is_output_step(int step) {
  return this->output_freq > 0 && (step % this->output_freq) == 0;
}

void PartSysBase::read_params() {
  // Output frequency
  // ToDo Should probably be unsigned, but complicates use of LoadParameter
  this->config->load_parameter(PART_OUTPUT_FREQ_STR, this->output_freq, 0);
  report_param("Output frequency (steps)", this->output_freq);
}

void PartSysBase::write(const int step) {
  if (this->h5part) {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle properties at step", step);
    }
    this->h5part->write();
  } else {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Ignoring call to write particle data because an output file "
             "wasn't set up. init_output() not called?");
    }
  }
};

void PartSysBase::init_object() {
  this->config->load_parameter(PART_OUTPUT_FREQ_STR, this->output_freq, 0);
  report_param("Output frequency (steps)", this->output_freq);

  // Create ParticleSpec
  this->init_spec();
  this->read_params();
  // Create ParticleGroup
  this->particle_group = std::make_shared<ParticleGroup>(
      this->domain, this->particle_spec, this->sycl_target);
  this->cell_id_translation = std::make_shared<CellIDTranslation>(
      this->sycl_target, this->particle_group->cell_id_dat,
      this->particle_mesh_interface);

  this->set_up_species();
}

} // namespace NESO::Particles
