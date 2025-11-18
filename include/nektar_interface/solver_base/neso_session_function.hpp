#ifndef NESO_SESSIONFUNCTION_H
#define NESO_SESSIONFUNCTION_H

#include "../../../include/nektar_interface/solver_base/neso_reader.hpp"

#include <LibUtilities/BasicUtils/CsvIO.h>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/Progressbar.hpp>
#include <LibUtilities/BasicUtils/PtsField.h>
#include <LibUtilities/BasicUtils/PtsIO.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <SolverUtils/Core/SessionFunction.h>
#include <MultiRegions/ExpList.h>
#include <SolverUtils/SolverUtilsDeclspec.h>

namespace MR = Nektar::MultiRegions;
using namespace Nektar;

namespace NESO {

class NESOSessionFunction {
public:
  /// Representation of a FUNCTION defined in the session xml file.
  NESOSessionFunction(const std::string&, const NESOReaderSharedPtr &session,
                      const MR::ExpListSharedPtr &field,
                      std::string functionName, bool toCache = false);

  /// Evaluates a function defined in the xml session file at each quadrature
  /// point.
  void Evaluate(Array<OneD, Array<OneD, NekDouble>> &pArray,
                const NekDouble pTime = 0.0, const int domain = 0);

  /// Evaluates a function defined in the xml session file at each quadrature
  /// point.
  void Evaluate(std::vector<std::string> pFieldNames,
                Array<OneD, Array<OneD, NekDouble>> &pArray,
                const NekDouble &pTime = 0.0, const int domain = 0);

  /// Evaluates a function defined in the xml session file at each quadrature
  /// point.
  void Evaluate(std::vector<std::string> pFieldNames,
                Array<OneD, MR::ExpListSharedPtr> &pFields,
                const NekDouble &pTime = 0.0, const int domain = 0);

  // Evaluates a function defined in the xml session file at each quadrature
  // point.
  void Evaluate(std::string pFieldName, Array<OneD, NekDouble> &pArray,
                const NekDouble &pTime = 0.0, const int domain = 0);

  /// Provide a description of a function for a given field name.
  std::string Describe(std::string pFieldName, const int domain = 0);

  const NESOReaderSharedPtr &GetSession() { return m_session; }

  const MR::ExpListSharedPtr &GetExpansion() { return m_field; }

private:
  /// Species Index
  std::string m_s;
  /// The session reader
  NESOReaderSharedPtr m_session;
  /// The expansion we want to evaluate this function for
  MR::ExpListSharedPtr m_field;
  // Name of this function
  std::string m_name;
  /// Store resulting arrays (and interpolators)
  bool m_toCache;
  /// Last time the cache for this variable & domain combo was updated
  std::map<std::pair<std::string, int>, NekDouble> m_lastCached;
  /// Interpolator for pts file input for a variable & domain combination
  std::map<std::string,
           FieldUtils::Interpolator<std::vector<MR::ExpListSharedPtr>>>
      m_interpolators;
  /// Cached result arrays
  std::map<std::pair<std::string, int>, Array<OneD, NekDouble>> m_arrays;

  // Evaluates a function from expression
  void EvaluateExp(std::string pFieldName, Array<OneD, NekDouble> &pArray,
                   const NekDouble &pTime = 0.0, const int domain = 0);

  // Evaluates a function from fld file
  void EvaluateFld(std::string pFieldName, Array<OneD, NekDouble> &pArray,
                   const NekDouble &pTime = 0.0, const int domain = 0);

  /// Evaluates a function from pts file
  void EvaluatePts(std::string pFieldName, Array<OneD, NekDouble> &pArray,
                   const NekDouble &pTime = 0.0, const int domain = 0);

  void PrintProgressbar(const int position, const int goal) const {
    LU::PrintProgressbar(position, goal, "Interpolating");
  }
};

typedef std::shared_ptr<NESOSessionFunction> NESOSessionFunctionSharedPtr;
static NESOSessionFunctionSharedPtr NullNESOSessionFunction;
} // namespace NESO

#endif
