///////////////////////////////////////////////////////////////////////////////
//
// File: neso_reader.hpp
// Based on nektar/library/LibUtilities/BasicUtils/SessionReader.hpp
// at https://gitlab.nektar.info/nektar by "Nektar++ developers"
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __NESO_READER_H_
#define __NESO_READER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SpatialDomains/Conditions.h>
#include <neso_particles/typedefs.hpp>
#include <optional>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
using Nektar::NekDouble;

namespace NESO {

// {parameters, functions, variables, (ion for neutral)}
typedef std::tuple<LU::ParameterMap, LU::FunctionMap, LU::VariableList,
                   std::optional<std::string>>
    SpeciesMap;
typedef std::map<std::string, SpeciesMap> SpeciesMapList;

enum class ParticleBoundaryConditionType {
  ePeriodic,
  eReflective,
  eNotDefined
};

typedef std::map<int, ParticleBoundaryConditionType>
    ParticleSpeciesBoundaryList;

enum class ParticleSourceType { ePoint, eBulk, eSurface };
// {number, type, (boundary), variables}
typedef std::tuple<int, ParticleSourceType, std::optional<int>,
                   LU::FunctionVariableMap>
    ParticleSource;

// {parameters, initial, sources, sinks, boundary, (ion for neutral)}
typedef std::tuple<LU::ParameterMap, std::pair<int, LU::FunctionVariableMap>,
                   std::vector<ParticleSource>,
                   std::vector<LU::FunctionVariableMap>,
                   ParticleSpeciesBoundaryList, std::optional<std::string>>
    ParticleSpeciesMap;
typedef std::map<std::string, ParticleSpeciesMap> ParticleSpeciesMapList;

// {name, species names, rates, cross-sections}
typedef std::tuple<std::string, std::vector<std::string>,
                   std::pair<std::string, NekDouble>,
                   std::pair<std::string, NekDouble>>
    ReactionMap;
typedef std::vector<ReactionMap> ReactionMapList;

// {name, species names, boundary ids, rates}
typedef std::tuple<std::string, std::vector<std::string>, std::vector<int>,
                   std::pair<std::string, NekDouble>>
    SurfaceReactionMap;
typedef std::vector<SurfaceReactionMap> SurfaceReactionMapList;

class NESOReader;
typedef std::shared_ptr<NESOReader> NESOReaderSharedPtr;

class NESOReader {
  friend class NESOSessionFunction;

public:
  NESOReader(const LU::SessionReaderSharedPtr session)
      : session(session), interpreter(session->GetInterpreter()) {};

  void read_species();

  inline const SpeciesMapList &get_species() const { return this->species; }

  void read_boundary_regions();

  std::map<int, std::vector<int>> &get_boundary_regions();

  /// @brief Reads the VANTAGE tag from xml document
  void read_vantage();

  std::vector<std::string> get_species_variables(const std::string &) const;
  void load_species_parameter(const std::string &, const std::string &name,
                              int &var) const;
  void load_species_parameter(const std::string &, const std::string &name,
                              int &var, const int &def) const;
  void load_species_parameter(const std::string &, const std::string &name,
                              NekDouble &var) const;
  void load_species_parameter(const std::string &, const std::string &name,
                              NekDouble &var, const NekDouble &def) const;
  bool defines_species_function(const std::string &,
                                const std::string &name) const;

  LU::EquationSharedPtr get_species_function(const std::string &,
                                             const std::string &name,
                                             const std::string &variable,
                                             const int pDomain = 0) const;
  enum LU::FunctionType get_species_function_type(const std::string &,
                                                  const std::string &name,
                                                  const std::string &variable,
                                                  const int pDomain = 0) const;
  /// Returns the type of a given function variable index.
  enum LU::FunctionType get_species_function_type(const std::string &,
                                                  const std::string &pName,
                                                  const unsigned int &pVar,
                                                  const int pDomain = 0) const;
  /// Returns the filename to be loaded for a given variable.
  std::string get_species_function_filename(const std::string &,
                                            const std::string &name,
                                            const std::string &variable,
                                            const int pDomain = 0) const;
  /// Returns the filename to be loaded for a given variable index.
  std::string get_species_function_filename(const std::string &,
                                            const std::string &name,
                                            const unsigned int &var,
                                            const int pDomain = 0) const;
  /// Returns the filename variable to be loaded for a given variable
  /// index.
  std::string get_species_function_filename_variable(
      const std::string &, const std::string &name, const std::string &variable,
      const int pDomain = 0) const;
  void read_species_functions(TiXmlElement *specie, LU::FunctionMap &map);

  void read_particle_species_initial(
      TiXmlElement *specie, std::pair<int, LU::FunctionVariableMap> &initial);
  void read_particle_species_sources(TiXmlElement *specie,
                                     std::vector<ParticleSource> &sources);
  void read_particle_species_sinks(TiXmlElement *specie,
                                   std::vector<LU::FunctionVariableMap> &sinks);
  void read_particle_species_boundary(TiXmlElement *specie,
                                      ParticleSpeciesBoundaryList &boundary);

  inline const ParticleSpeciesMapList &get_particle_species() const {
    return this->particle_species;
  }

  void read_reactions(TiXmlElement *vantage);
  inline const ReactionMapList &get_reactions() const {
    return this->reactions;
  }
  void read_surface_reactions(TiXmlElement *vantage);
  inline const SurfaceReactionMapList &get_surface_reactions() const {
    return this->surface_reactions;
  }

  void load_particle_species_parameter(const std::string &s,
                                       const std::string &name, int &var) const;
  void load_particle_species_parameter(const std::string &s,
                                       const std::string &name, int &var,
                                       const int &def) const;

  void load_particle_species_parameter(const std::string &s,
                                       const std::string &name,
                                       NekDouble &var) const;
  void load_particle_species_parameter(const std::string &s,
                                       const std::string &name, NekDouble &var,
                                       const NekDouble &def) const;

  int get_particle_species_initial_N(const std::string &s) const;

  /// Returns an EquationSharedPtr to a given function variable.
  LU::FunctionVariableMap
  get_particle_species_initial(const std::string &s) const;

  const std::vector<ParticleSource> &
  get_particle_species_sources(const std::string &s) const;

  const std::vector<LU::FunctionVariableMap> &
  get_particle_species_sinks(const std::string &s) const;

  const ParticleSpeciesBoundaryList &
  get_particle_species_boundary(const std::string &s) const;

  /// Load an integer parameter
  void load_parameter(const std::string &name, int &var) const;
  /// Load an size_t parameter
  void load_parameter(const std::string &name, size_t &var) const;
  /// Check for and load an integer parameter.
  void load_parameter(const std::string &name, int &var, const int &def) const;
  /// Check for and load an size_t parameter.
  void load_parameter(const std::string &name, size_t &var,
                      const size_t &def) const;
  /// Load a double precision parameter
  void load_parameter(const std::string &name, NekDouble &var) const;
  /// Check for and load a double-precision parameter.
  void load_parameter(const std::string &name, NekDouble &var,
                      const NekDouble &def) const;
  void read_species_parameter(TiXmlElement *specie, LU::ParameterMap &map);

private:
  // Nektar++ SessionReader
  LU::SessionReaderSharedPtr session;
  // Expression interptreter
  LU::InterpreterSharedPtr interpreter;
  // Map of particle species
  SpeciesMapList species;
  /// Map of particle info (e.g. Particle System name)
  std::map<std::string, std::string> particle_info;
  // Map of particle species
  ParticleSpeciesMapList particle_species;
  // Particle parameters
  LU::ParameterMap parameters;
  /// Functions.
  LU::FunctionMap functions;

  // Reactions
  ReactionMapList reactions;
  SurfaceReactionMapList surface_reactions;

  std::map<int, std::vector<int>> boundary_groups;

  void parse_equals(const std::string &line, std::string &lhs,
                    std::string &rhs);
};

} // namespace NESO
#endif