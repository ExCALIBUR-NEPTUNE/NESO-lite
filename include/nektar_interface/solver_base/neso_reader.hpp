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

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
using Nektar::NekDouble;

namespace NESO {

// {name, parameters, functions, variables}
typedef std::tuple<std::string, LU::ParameterMap, LU::FunctionMap,
                   LU::VariableList>
    SpeciesMap;
typedef std::map<int, SpeciesMap> SpeciesMapList;

enum class ParticleBoundaryConditionType {
  ePeriodic,
  eReflective,
  eNotDefined
};

typedef std::map<int, ParticleBoundaryConditionType>
    ParticleSpeciesBoundaryList;

// {name, parameters, initial, sources, sinks, boundary}
typedef std::tuple<
    std::string, LU::ParameterMap, std::pair<int, LU::FunctionVariableMap>,
    std::vector<std::pair<int, LU::FunctionVariableMap>>,
    std::vector<LU::FunctionVariableMap>, ParticleSpeciesBoundaryList>
    ParticleSpeciesMap;
typedef std::map<int, ParticleSpeciesMap> ParticleSpeciesMapList;

typedef std::tuple<std::string, std::vector<int>,
                   std::pair<std::string, NekDouble>,
                   std::pair<std::string, NekDouble>>
    ReactionMap;
typedef std::map<int, ReactionMap> ReactionMapList;
typedef std::tuple<std::string, std::vector<int>, std::vector<int>,
                   std::pair<std::string, NekDouble>>
    SurfaceReactionMap;
typedef std::map<int, SurfaceReactionMap> SurfaceReactionMapList;

class NESOReader;
typedef std::shared_ptr<NESOReader> NESOReaderSharedPtr;

class NESOReader {
  friend class NESOSessionFunction;

public:
  NESOReader(const LU::SessionReaderSharedPtr session)
      : session(session), interpreter(session->GetInterpreter()) {};

  void read_species();

  inline const SpeciesMapList &get_species() const { return this->species; }

  /// @brief Reads the particle tag from xml document
  void read_particles();

  /// @brief Reads info related to particles
  void read_info();
  /// Checks if info is specified in the XML document.
  bool defines_info(const std::string &name) const;
  /// Returns the value of the particle info.
  const std::string &get_info(const std::string &name) const;

  /// @brief  Reads parameters related to particles
  /// @param particles
  void read_parameters(TiXmlElement *particles);
  /// Checks if a parameter is specified in the XML document.
  bool defines_parameter(const std::string &name) const;
  /// Returns the value of the specified parameter.
  const NekDouble &get_parameter(const std::string &name) const;
  std::vector<std::string> get_species_variables(int s) const;
  void load_species_parameter(const int s, const std::string &name,
                              int &var) const;

  void load_species_parameter(const int s, const std::string &name,
                              NekDouble &var) const;
  bool defines_species_function(int s, const std::string &name) const;

  LU::EquationSharedPtr get_species_function(int s, const std::string &name,
                                             const std::string &variable,
                                             const int pDomain = 0) const;
  enum LU::FunctionType get_species_function_type(int s,
                                                  const std::string &name,
                                                  const std::string &variable,
                                                  const int pDomain = 0) const;
  /// Returns the type of a given function variable index.
  enum LU::FunctionType get_species_function_type(int s,
                                                  const std::string &pName,
                                                  const unsigned int &pVar,
                                                  const int pDomain = 0) const;
  /// Returns the filename to be loaded for a given variable.
  std::string get_species_function_filename(int s, const std::string &name,
                                            const std::string &variable,
                                            const int pDomain = 0) const;
  /// Returns the filename to be loaded for a given variable index.
  std::string get_species_function_filename(int s, const std::string &name,
                                            const unsigned int &var,
                                            const int pDomain = 0) const;
  /// Returns the filename variable to be loaded for a given variable
  /// index.
  std::string
  get_species_function_filename_variable(int s, const std::string &name,
                                         const std::string &variable,
                                         const int pDomain = 0) const;
  void read_species_functions(TiXmlElement *specie, LU::FunctionMap &map);

  /// @brief  Reads initial conditions for a species
  /// @param particles
  /// @param initial
  void read_particle_species_initial(
      TiXmlElement *particles,
      std::pair<int, LU::FunctionVariableMap> &initial);
  /// @brief  Reads the sources defined for a species
  /// @param particles
  /// @param sources
  void read_particle_species_sources(
      TiXmlElement *particles,
      std::vector<std::pair<int, LU::FunctionVariableMap>> &sources);

  /// @brief  Reads the sinks defined for a species
  /// @param particles
  /// @param sinks
  void read_particle_species_sinks(TiXmlElement *particles,
                                   std::vector<LU::FunctionVariableMap> &sinks);
  void read_particle_species_boundary(TiXmlElement *specie,
                                      ParticleSpeciesBoundaryList &boundary);

  /// @brief Reads the list of species defined under particles
  /// @param particles
  void read_particle_species(TiXmlElement *particles);

  inline const ParticleSpeciesMapList &get_particle_species() const {
    return this->particle_species;
  }

  /// @brief Reads reactions
  /// @param particles
  void read_reactions(TiXmlElement *particles);
  inline const ReactionMapList &get_reactions() const {
    return this->reactions;
  }

  /// @brief Reads surface reactions
  /// @param particles
  void read_surface_reactions(TiXmlElement *particles);
  inline const SurfaceReactionMapList &get_surface_reactions() const {
    return this->surface_reactions;
  }

  /// @param species
  /// @param name
  /// @param var
  void load_particle_species_parameter(const int species,
                                       const std::string &name, int &var) const;
  /// @brief Loads a species parameter (double)
  /// @param species
  /// @param name
  /// @param var
  void load_particle_species_parameter(const int species,
                                       const std::string &name,
                                       NekDouble &var) const;

  int get_particle_species_initial_N(const int species) const;

  /// Returns an EquationSharedPtr to a given function variable.
  LU::EquationSharedPtr
  get_particle_species_initial(const int species, const std::string &variable,
                               const int pDomain = 0) const;
  /// Returns an EquationSharedPtr to a given function variable index.
  LU::EquationSharedPtr
  get_particle_species_initial(const int species, const unsigned int &var,
                               const int pDomain = 0) const;

  const std::vector<std::pair<int, LU::FunctionVariableMap>> &
  get_particle_species_sources(const int species) const;

  const std::vector<LU::FunctionVariableMap> &
  get_particle_species_sinks(const int species) const;

  const ParticleSpeciesBoundaryList &
  get_particle_species_boundary(const int species) const;

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

  void parse_equals(const std::string &line, std::string &lhs,
                    std::string &rhs);
};

} // namespace NESO
#endif