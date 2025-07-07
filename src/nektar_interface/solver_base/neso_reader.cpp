///////////////////////////////////////////////////////////////////////////////
//
// File: neso_reader.cpp
// Based on nektar/library/LibUtilities/BasicUtils/SessionReader.cpp
// at https://gitlab.nektar.info/nektar by "Nektar++ developers"
//
///////////////////////////////////////////////////////////////////////////////

#include "../../../include/nektar_interface/solver_base/neso_reader.hpp"
#include <fstream>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <tinyxml.h>

#include <LibUtilities/BasicUtils/CheckedCast.hpp>
#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

using Nektar::ParseUtils;
template <typename DataType>
using MemoryManager = Nektar::MemoryManager<DataType>;

namespace NESO {
/**
 *
 */
void NESOReader::parse_equals(const std::string &line, std::string &lhs,
                              std::string &rhs) {
  /// Pull out lhs and rhs and eliminate any spaces.
  size_t beg = line.find_first_not_of(" ");
  size_t end = line.find_first_of("=");
  // Check for no parameter name
  if (beg == end) {
    throw 1;
  }
  // Check for no parameter value
  if (end != line.find_last_of("=")) {
    throw 1;
  }
  // Check for no equals sign
  if (end == std::string::npos) {
    throw 1;
  }

  lhs = line.substr(line.find_first_not_of(" "), end - beg);
  lhs = lhs.substr(0, lhs.find_last_not_of(" ") + 1);
  rhs = line.substr(line.find_last_of("=") + 1);
  rhs = rhs.substr(rhs.find_first_not_of(" "));
  rhs = rhs.substr(0, rhs.find_last_not_of(" ") + 1);
}

void NESOReader::read_species() {
  NESOASSERT(&this->session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&this->session->GetDocument());
  TiXmlElement *species;

  // Look for all data in PARTICLES block.
  species = docHandle.FirstChildElement("NEKTAR")
                .FirstChildElement("NESO")
                .FirstChildElement("SPECIES")
                .Element();
  if (species) {
    TiXmlElement *specie = species->FirstChildElement("S");

    while (specie) {
      std::stringstream tagcontent;
      tagcontent << *specie;
      std::string id = specie->Attribute("ID");
      NESOASSERT(!id.empty(), "Missing ID attribute in Species XML "
                              "element: \n\t'" +
                                  tagcontent.str() + "'");

      std::string name = specie->Attribute("NAME");
      NESOASSERT(!name.empty(),
                 "NAME attribute must be non-empty in XML element:\n\t'" +
                     tagcontent.str() + "'");
      SpeciesMap species_map;
      std::get<0>(species_map) = name;

      TiXmlElement *parameter = specie->FirstChildElement("P");

      // Multiple nodes will only occur if there is a comment in
      // between definitions.
      while (parameter) {
        std::stringstream tagcontent;
        tagcontent << *parameter;
        TiXmlNode *node = parameter->FirstChild();

        while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
          node = node->NextSibling();
        }

        if (node) {
          // Format is "paramName = value"
          std::string line = node->ToText()->Value(), lhs, rhs;

          try {
            parse_equals(line, lhs, rhs);
          } catch (...) {
            NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                  "' in XML element: \n\t'" + tagcontent.str() +
                                  "'");
          }

          // We want the list of parameters to have their RHS
          // evaluated, so we use the expression evaluator to do
          // the dirty work.
          if (!lhs.empty() && !rhs.empty()) {
            NekDouble value = 0.0;
            try {
              LU::Equation expession(this->interpreter, rhs);
              value = expession.Evaluate();
            } catch (const std::runtime_error &) {
              NESOASSERT(false, "Error evaluating parameter expression"
                                " '" +
                                    rhs + "' in XML element: \n\t'" +
                                    tagcontent.str() + "'");
            }
            this->interpreter->SetParameter(lhs, value);
            boost::to_upper(lhs);
            std::get<1>(species_map)[lhs] = value;
          }
        }
        parameter = parameter->NextSiblingElement();
      }
      read_species_functions(specie, std::get<2>(species_map));
      specie = specie->NextSiblingElement("S");

      this->species[std::stoi(id)] = species_map;
    }
  }
}

void NESOReader::read_info() {
  NESOASSERT(&this->session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&this->session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("NESO")
                  .FirstChildElement("PARTICLES")
                  .Element();
  if (!particles) {
    return;
  }
  this->particle_info.clear();

  TiXmlElement *particle_info_element = particles->FirstChildElement("INFO");

  if (particle_info_element) {
    TiXmlElement *particle_info_i =
        particle_info_element->FirstChildElement("I");

    while (particle_info_i) {
      std::stringstream tagcontent;
      tagcontent << *particle_info_i;
      // read the property name
      NESOASSERT(particle_info_i->Attribute("PROPERTY"),
                 "Missing PROPERTY attribute in particle info "
                 "XML element: \n\t'" +
                     tagcontent.str() + "'");
      std::string particle_property = particle_info_i->Attribute("PROPERTY");
      NESOASSERT(!particle_property.empty(),
                 "PROPERTY attribute must be non-empty in XML "
                 "element: \n\t'" +
                     tagcontent.str() + "'");

      // make sure that solver property is capitalised
      std::string particle_property_upper =
          boost::to_upper_copy(particle_property);

      // read the value
      NESOASSERT(particle_info_i->Attribute("VALUE"),
                 "Missing VALUE attribute in particle info "
                 "XML element: \n\t'" +
                     tagcontent.str() + "'");
      std::string particle_value = particle_info_i->Attribute("VALUE");
      NESOASSERT(!particle_value.empty(),
                 "VALUE attribute must be non-empty in XML "
                 "element: \n\t'" +
                     tagcontent.str() + "'");

      // Set Variable
      this->particle_info[particle_property_upper] = particle_value;
      particle_info_i = particle_info_i->NextSiblingElement("I");
    }
  }
}

/**
 *
 */
bool NESOReader::defines_info(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  return this->particle_info.find(name_upper) != this->particle_info.end();
}
/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   name       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const std::string &NESOReader::get_info(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto info_iter = this->particle_info.find(name);

  NESOASSERT(info_iter != this->particle_info.end(),
             "Unable to find requested info: " + name);

  return info_iter->second;
}

void NESOReader::read_parameters(TiXmlElement *particles) {
  this->parameters.clear();

  TiXmlElement *parameters = particles->FirstChildElement("PARAMETERS");

  // See if we have parameters defined.  They are optional so we go on
  // if not.
  if (parameters) {
    TiXmlElement *parameter_p = parameters->FirstChildElement("P");

    // Multiple nodes will only occur if there is a comment in
    // between definitions.
    while (parameter_p) {
      std::stringstream tagcontent;
      tagcontent << *parameter_p;
      TiXmlNode *node = parameter_p->FirstChild();

      while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
        node = node->NextSibling();
      }

      if (node) {
        // Format is "paramName = value"
        std::string line = node->ToText()->Value(), lhs, rhs;

        try {
          parse_equals(line, lhs, rhs);
        } catch (...) {
          NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                "' in XML element: \n\t'" + tagcontent.str() +
                                "'");
        }

        // We want the list of parameters to have their RHS
        // evaluated, so we use the expression evaluator to do
        // the dirty work.
        if (!lhs.empty() && !rhs.empty()) {
          NekDouble value = 0.0;
          try {
            LU::Equation expession(this->interpreter, rhs);
            value = expession.Evaluate();
          } catch (const std::runtime_error &) {
            NESOASSERT(false, "Error evaluating parameter expression"
                              " '" +
                                  rhs + "' in XML element: \n\t'" +
                                  tagcontent.str() + "'");
          }
          this->interpreter->SetParameter(lhs, value);
          boost::to_upper(lhs);
          this->parameters[lhs] = value;
        }
      }
      parameter_p = parameter_p->NextSiblingElement();
    }
  }
}

/**
 *
 */
bool NESOReader::defines_parameter(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  return this->parameters.find(name_upper) != this->parameters.end();
}

/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   name       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const NekDouble &NESOReader::get_parameter(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);

  NESOASSERT(param_iter != this->parameters.end(),
             "Unable to find requested parameter: " + name);

  return param_iter->second;
}

std::vector<std::string> NESOReader::get_species_variables(int s) const {}
bool NESOReader::defines_species_function(int s,
                                          const std::string &name) const {
  auto functions = std::get<2>(this->species.at(s));
  std::string vName = boost::to_upper_copy(name);
  return functions.find(vName) != functions.end();
}

LU::EquationSharedPtr
NESOReader::get_species_function(int s, const std::string &name,
                                 const std::string &variable,
                                 const int pDomain) const {
  std::string vName = boost::to_upper_copy(name);
  auto functions = std::get<2>(this->species.at(s));
  auto it1 = functions.find(vName);

  ASSERTL0(it1 != functions.end(),
           std::string("No such function '") + name +
               std::string("' has been defined in the session file."));

  // Check for specific and wildcard definitions
  std::pair<std::string, int> key(variable, pDomain);
  std::pair<std::string, int> defkey("*", pDomain);

  auto it2 = it1->second.find(key);
  auto it3 = it1->second.find(defkey);
  bool specific = it2 != it1->second.end();
  bool wildcard = it3 != it1->second.end();

  // Check function is defined somewhere
  ASSERTL0(specific || wildcard, "No such variable " + variable +
                                     " in domain " + std::to_string(pDomain) +
                                     " defined for function " + name +
                                     " in session file.");

  // If not specific, must be wildcard
  if (!specific) {
    it2 = it3;
  }

  ASSERTL0((it2->second.m_type == LU::eFunctionTypeExpression),
           std::string("Function is defined by a file."));
  return it2->second.m_expression;
}

enum LU::FunctionType
NESOReader::get_species_function_type(int s, const std::string &name,
                                      const std::string &variable,
                                      const int pDomain) const {
  std::string vName = boost::to_upper_copy(name);
  auto functions = std::get<2>(this->species.at(s));
  auto it1 = functions.find(vName);

  ASSERTL0(it1 != functions.end(),
           std::string("Function '") + name + std::string("' not found."));

  // Check for specific and wildcard definitions
  std::pair<std::string, int> key(variable, pDomain);
  std::pair<std::string, int> defkey("*", pDomain);

  auto it2 = it1->second.find(key);
  auto it3 = it1->second.find(defkey);
  bool specific = it2 != it1->second.end();
  bool wildcard = it3 != it1->second.end();

  // Check function is defined somewhere
  ASSERTL0(specific || wildcard, "No such variable " + variable +
                                     " in domain " + std::to_string(pDomain) +
                                     " defined for function " + name +
                                     " in session file.");

  // If not specific, must be wildcard
  if (!specific) {
    it2 = it3;
  }

  return it2->second.m_type;
}

/// Returns the filename to be loaded for a given variable.
std::string
NESOReader::get_species_function_filename(int s, const std::string &name,
                                          const std::string &variable,
                                          const int pDomain) const {
  std::string vName = boost::to_upper_copy(name);
  auto functions = std::get<2>(this->species.at(s));
  auto it1 = functions.find(vName);

  ASSERTL0(it1 != functions.end(),
           std::string("Function '") + name + std::string("' not found."));

  // Check for specific and wildcard definitions
  std::pair<std::string, int> key(variable, pDomain);
  std::pair<std::string, int> defkey("*", pDomain);

  auto it2 = it1->second.find(key);
  auto it3 = it1->second.find(defkey);
  bool specific = it2 != it1->second.end();
  bool wildcard = it3 != it1->second.end();

  // Check function is defined somewhere
  ASSERTL0(specific || wildcard, "No such variable " + variable +
                                     " in domain " + std::to_string(pDomain) +
                                     " defined for function " + name +
                                     " in session file.");

  // If not specific, must be wildcard
  if (!specific) {
    it2 = it3;
  }

  return it2->second.m_filename;
}

std::string NESOReader::get_species_function_filename_variable(
    int s, const std::string &name, const std::string &variable,
    const int pDomain) const {
  std::string vName = boost::to_upper_copy(name);
  auto functions = std::get<2>(this->species.at(s));

  auto it1 = functions.find(vName);

  ASSERTL0(it1 != functions.end(),
           std::string("Function '") + name + std::string("' not found."));

  // Check for specific and wildcard definitions
  std::pair<std::string, int> key(variable, pDomain);
  std::pair<std::string, int> defkey("*", pDomain);

  auto it2 = it1->second.find(key);
  auto it3 = it1->second.find(defkey);
  bool specific = it2 != it1->second.end();
  bool wildcard = it3 != it1->second.end();

  // Check function is defined somewhere
  ASSERTL0(specific || wildcard, "No such variable " + variable +
                                     " in domain " + std::to_string(pDomain) +
                                     " defined for function " + name +
                                     " in session file.");

  // If not specific, must be wildcard
  if (!specific) {
    it2 = it3;
  }

  return it2->second.m_fileVariable;
}

void NESOReader::read_species_functions(TiXmlElement *specie,
                                        LU::FunctionMap &map) {
  map.clear();

  if (!specie) {
    return;
  }

  // Scan through conditions section looking for functions.
  TiXmlElement *function = specie->FirstChildElement("FUNCTION");

  while (function) {
    std::stringstream tagcontent;
    tagcontent << *function;

    // Every function must have a NAME attribute
    NESOASSERT(function->Attribute("NAME"),
               "Functions must have a NAME attribute defined in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");
    std::string functionStr = function->Attribute("NAME");
    NESOASSERT(!functionStr.empty(),
               "Functions must have a non-empty name in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");

    // Store function names in uppercase to remain case-insensitive.
    boost::to_upper(functionStr);

    // Retrieve first entry (variable, or file)
    TiXmlElement *element = function;
    TiXmlElement *variable = element->FirstChildElement();

    // Create new function structure with default type of none.
    LU::FunctionVariableMap functionVarMap;

    // Process all entries in the function block
    while (variable) {
      LU::FunctionVariableDefinition funcDef;
      std::string conditionType = variable->Value();

      // If no var is specified, assume wildcard
      std::string variableStr;
      if (!variable->Attribute("VAR")) {
        variableStr = "*";
      } else {
        variableStr = variable->Attribute("VAR");
      }

      // Parse list of variables
      std::vector<std::string> variableList;
      ParseUtils::GenerateVector(variableStr, variableList);

      // If no domain is specified, put to 0
      std::string domainStr;
      if (!variable->Attribute("DOMAIN")) {
        domainStr = "0";
      } else {
        domainStr = variable->Attribute("DOMAIN");
      }

      // Parse list of domains
      std::vector<std::string> varSplit;
      std::vector<unsigned int> domainList;
      ParseUtils::GenerateSeqVector(domainStr, domainList);

      // if no evars is specified, put "x y z t"
      std::string evarsStr = "x y z t";
      if (variable->Attribute("EVARS")) {
        evarsStr = evarsStr + std::string(" ") + variable->Attribute("EVARS");
      }

      // Expressions are denoted by E
      if (conditionType == "E") {
        funcDef.m_type = LU::eFunctionTypeExpression;

        // Expression must have a VALUE.
        NESOASSERT(variable->Attribute("VALUE"),
                   "Attribute VALUE expected for function '" + functionStr +
                       "'.");
        std::string fcnStr = variable->Attribute("VALUE");
        NESOASSERT(!fcnStr.empty(),
                   (std::string("Expression for var: ") + variableStr +
                    std::string(" must be specified."))
                       .c_str());

        // set expression
        funcDef.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
            this->interpreter, fcnStr, evarsStr);
      }

      // Files are denoted by F
      else if (conditionType == "F") {
        // Check if transient or not
        if (variable->Attribute("TIMEDEPENDENT") &&
            boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
          funcDef.m_type = LU::eFunctionTypeTransientFile;
        } else {
          funcDef.m_type = LU::eFunctionTypeFile;
        }

        // File must have a FILE.
        NESOASSERT(variable->Attribute("FILE"),
                   "Attribute FILE expected for function '" + functionStr +
                       "'.");
        std::string filenameStr = variable->Attribute("FILE");
        NESOASSERT(!filenameStr.empty(),
                   "A filename must be specified for the FILE "
                   "attribute of function '" +
                       functionStr + "'.");

        std::vector<std::string> fSplit;
        boost::split(fSplit, filenameStr, boost::is_any_of(":"));
        NESOASSERT(fSplit.size() == 1 || fSplit.size() == 2,
                   "Incorrect filename specification in function " +
                       functionStr +
                       "'. "
                       "Specify variables inside file as: "
                       "filename:var1,var2");

        // set the filename
        fs::path fullpath = fSplit[0];
        fs::path ftype = fullpath.extension();

        if (fSplit.size() == 2) {
          NESOASSERT(variableList[0] != "*",
                     "Filename variable mapping not valid "
                     "when using * as a variable inside "
                     "function '" +
                         functionStr + "'.");

          boost::split(varSplit, fSplit[1], boost::is_any_of(","));
          NESOASSERT(varSplit.size() == variableList.size(),
                     "Filename variables should contain the "
                     "same number of variables defined in "
                     "VAR in function " +
                         functionStr + "'.");
        }
      }

      // Nothing else supported so throw an error
      else {
        std::stringstream tagcontent;
        tagcontent << *variable;

        NESOASSERT(false, "Identifier " + conditionType + " in function " +
                              std::string(function->Attribute("NAME")) +
                              " is not recognised in XML element: \n\t'" +
                              tagcontent.str() + "'");
      }

      // Add variables to function
      for (unsigned int i = 0; i < variableList.size(); ++i) {
        for (unsigned int j = 0; j < domainList.size(); ++j) {
          // Check it has not already been defined
          std::pair<std::string, int> key(variableList[i], domainList[j]);
          auto fcnsIter = functionVarMap.find(key);
          NESOASSERT(fcnsIter == functionVarMap.end(),
                     "Error setting expression '" + variableList[i] +
                         " in domain " + std::to_string(domainList[j]) +
                         "' in function '" + functionStr +
                         "'. "
                         "Expression has already been defined.");

          if (varSplit.size() > 0) {
            LU::FunctionVariableDefinition funcDef2 = funcDef;
            funcDef2.m_fileVariable = varSplit[i];
            functionVarMap[key] = funcDef2;
          } else {
            functionVarMap[key] = funcDef;
          }
        }
      }
      variable = variable->NextSiblingElement();
    }

    // Add function definition to map
    map[functionStr] = functionVarMap;
    function = function->NextSiblingElement("FUNCTION");
  }
}

void NESOReader::read_particle_species_sources(
    TiXmlElement *specie,
    std::vector<std::pair<int, LU::FunctionVariableMap>> &sources) {
  if (!specie) {
    return;
  }

  // Scan through conditions section looking for functions.
  TiXmlElement *function = specie->FirstChildElement("SOURCE");

  while (function) {
    std::stringstream tagcontent;
    tagcontent << *function;

    // Every function must have a NAME attribute
    NESOASSERT(function->Attribute("N"),
               "Sources must have N attribute defined in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");
    std::string N_str = function->Attribute("N");
    NESOASSERT(!N_str.empty(), "Sources must have a non-empty N in XML "
                               "element: \n\t'" +
                                   tagcontent.str() + "'");

    int N = std::stoi(N_str);

    // Retrieve first entry (variable, or file)
    TiXmlElement *element = function;
    TiXmlElement *variable = element->FirstChildElement();

    // Create new function structure with default type of none.
    LU::FunctionVariableMap function_var_map;

    // Process all entries in the function block
    while (variable) {
      LU::FunctionVariableDefinition func_def;
      std::string condition_type = variable->Value();

      // If no var is specified, assume wildcard
      std::string variable_str;
      if (!variable->Attribute("VAR")) {
        variable_str = "*";
      } else {
        variable_str = variable->Attribute("VAR");
      }

      // Parse list of variables
      std::vector<std::string> variable_list;
      ParseUtils::GenerateVector(variable_str, variable_list);

      // If no domain is specified, put to 0
      std::string domain_str;
      if (!variable->Attribute("DOMAIN")) {
        domain_str = "0";
      } else {
        domain_str = variable->Attribute("DOMAIN");
      }

      // Parse list of domains
      std::vector<std::string> var_split;
      std::vector<unsigned int> domain_list;
      ParseUtils::GenerateSeqVector(domain_str, domain_list);

      // if no evars is specified, put "x y z t"
      std::string evars_str = "x y z t";
      if (variable->Attribute("EVARS")) {
        evars_str = evars_str + std::string(" ") + variable->Attribute("EVARS");
      }

      // Expressions are denoted by E
      if (condition_type == "E") {
        func_def.m_type = LU::eFunctionTypeExpression;

        // Expression must have a VALUE.
        NESOASSERT(variable->Attribute("VALUE"),
                   "Attribute VALUE expected for SOURCE expression");
        std::string fcn_str = variable->Attribute("VALUE");
        NESOASSERT(!fcn_str.empty(),
                   (std::string("Expression for var: ") + variable_str +
                    std::string(" must be specified."))
                       .c_str());

        // set expression
        func_def.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
            this->interpreter, fcn_str, evars_str);
      }

      // Files are denoted by F
      else if (condition_type == "F") {
        // Check if transient or not
        if (variable->Attribute("TIMEDEPENDENT") &&
            boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
          func_def.m_type = LU::eFunctionTypeTransientFile;
        } else {
          func_def.m_type = LU::eFunctionTypeFile;
        }

        // File must have a FILE.
        NESOASSERT(variable->Attribute("FILE"),
                   "Attribute FILE expected for source.");
        std::string filename_str = variable->Attribute("FILE");
        NESOASSERT(!filename_str.empty(),
                   "A filename must be specified for the FILE "
                   "attribute of SOURCE.");

        std::vector<std::string> f_split;
        boost::split(f_split, filename_str, boost::is_any_of(":"));
        NESOASSERT(f_split.size() == 1 || f_split.size() == 2,
                   "Incorrect filename specification in SOURCE. "
                   "Specify variables inside file as: "
                   "filename:var1,var2");

        // set the filename
        fs::path fullpath = f_split[0];
        func_def.m_filename = fullpath.string();

        if (f_split.size() == 2) {
          NESOASSERT(variable_list[0] != "*",
                     "Filename variable mapping not valid "
                     "when using * as a variable inside "
                     "SOURCE.");

          boost::split(var_split, f_split[1], boost::is_any_of(","));
          NESOASSERT(var_split.size() == variable_list.size(),
                     "Filename variables should contain the "
                     "same number of variables defined in "
                     "VAR in SOURCE.");
        }
      }

      // Nothing else supported so throw an error
      else {
        std::stringstream tagcontent;
        tagcontent << *variable;

        NESOASSERT(false, "Identifier " + condition_type + " in function " +
                              std::string(function->Attribute("NAME")) +
                              " is not recognised in XML element: \n\t'" +
                              tagcontent.str() + "'");
      }

      // Add variables to function
      for (unsigned int i = 0; i < variable_list.size(); ++i) {
        for (unsigned int j = 0; j < domain_list.size(); ++j) {
          // Check it has not already been defined
          std::pair<std::string, int> key(variable_list[i], domain_list[j]);
          auto fcns_iter = function_var_map.find(key);
          NESOASSERT(fcns_iter == function_var_map.end(),
                     "Error setting expression '" + variable_list[i] +
                         " in domain " + std::to_string(domain_list[j]) +
                         "' in SOURCE. "
                         "Expression has already been defined.");

          if (var_split.size() > 0) {
            LU::FunctionVariableDefinition func_def2 = func_def;
            func_def2.m_fileVariable = var_split[i];
            function_var_map[key] = func_def2;
          } else {
            function_var_map[key] = func_def;
          }
        }
      }
      variable = variable->NextSiblingElement();
    }

    // Add function definition to map
    sources.push_back(std::make_pair(N, function_var_map));
    function = function->NextSiblingElement("SOURCE");
  }
}

void NESOReader::read_particle_species_sinks(
    TiXmlElement *specie, std::vector<LU::FunctionVariableMap> &sinks) {
  if (!specie) {
    return;
  }

  // Scan through conditions section looking for functions.
  TiXmlElement *function = specie->FirstChildElement("SINK");

  while (function) {
    std::stringstream tagcontent;
    tagcontent << *function;

    // Retrieve first entry (variable, or file)
    TiXmlElement *element = function;
    TiXmlElement *variable = element->FirstChildElement();

    // Create new function structure with default type of none.
    LU::FunctionVariableMap function_var_map;

    // Process all entries in the function block
    while (variable) {
      LU::FunctionVariableDefinition func_def;
      std::string condition_type = variable->Value();

      // If no var is specified, assume wildcard
      std::string variable_str;
      if (!variable->Attribute("VAR")) {
        variable_str = "*";
      } else {
        variable_str = variable->Attribute("VAR");
      }

      // Parse list of variables
      std::vector<std::string> variable_list;
      ParseUtils::GenerateVector(variable_str, variable_list);

      // If no domain is specified, put to 0
      std::string domain_str;
      if (!variable->Attribute("DOMAIN")) {
        domain_str = "0";
      } else {
        domain_str = variable->Attribute("DOMAIN");
      }

      // Parse list of domains
      std::vector<std::string> var_split;
      std::vector<unsigned int> domain_list;
      ParseUtils::GenerateSeqVector(domain_str, domain_list);

      // if no evars is specified, put "x y z t"
      std::string evars_str = "x y z t";
      if (variable->Attribute("EVARS")) {
        evars_str = evars_str + std::string(" ") + variable->Attribute("EVARS");
      }

      // Expressions are denoted by E
      if (condition_type == "E") {
        func_def.m_type = LU::eFunctionTypeExpression;

        // Expression must have a VALUE.
        NESOASSERT(variable->Attribute("VALUE"),
                   "Attribute VALUE expected for SINK expression");
        std::string fcn_str = variable->Attribute("VALUE");
        NESOASSERT(!fcn_str.empty(),
                   (std::string("Expression for var: ") + variable_str +
                    std::string(" must be specified."))
                       .c_str());

        // set expression
        func_def.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
            this->interpreter, fcn_str, evars_str);
      }

      // Files are denoted by F
      else if (condition_type == "F") {
        // Check if transient or not
        if (variable->Attribute("TIMEDEPENDENT") &&
            boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
          func_def.m_type = LU::eFunctionTypeTransientFile;
        } else {
          func_def.m_type = LU::eFunctionTypeFile;
        }

        // File must have a FILE.
        NESOASSERT(variable->Attribute("FILE"),
                   "Attribute FILE expected for sink.");
        std::string filename_str = variable->Attribute("FILE");
        NESOASSERT(!filename_str.empty(),
                   "A filename must be specified for the FILE "
                   "attribute of SINK.");

        std::vector<std::string> f_split;
        boost::split(f_split, filename_str, boost::is_any_of(":"));
        NESOASSERT(f_split.size() == 1 || f_split.size() == 2,
                   "Incorrect filename specification in SINK. "
                   "Specify variables inside file as: "
                   "filename:var1,var2");

        // set the filename
        fs::path fullpath = f_split[0];
        func_def.m_filename = fullpath.string();

        if (f_split.size() == 2) {
          NESOASSERT(variable_list[0] != "*",
                     "Filename variable mapping not valid "
                     "when using * as a variable inside "
                     "SINK.");

          boost::split(var_split, f_split[1], boost::is_any_of(","));
          NESOASSERT(var_split.size() == variable_list.size(),
                     "Filename variables should contain the "
                     "same number of variables defined in "
                     "VAR in SINK.");
        }
      }

      // Nothing else supported so throw an error
      else {
        std::stringstream tagcontent;
        tagcontent << *variable;

        NESOASSERT(false, "Identifier " + condition_type + " in function " +
                              std::string(function->Attribute("NAME")) +
                              " is not recognised in XML element: \n\t'" +
                              tagcontent.str() + "'");
      }

      // Add variables to function
      for (unsigned int i = 0; i < variable_list.size(); ++i) {
        for (unsigned int j = 0; j < domain_list.size(); ++j) {
          // Check it has not already been defined
          std::pair<std::string, int> key(variable_list[i], domain_list[j]);
          auto fcns_iter = function_var_map.find(key);
          NESOASSERT(fcns_iter == function_var_map.end(),
                     "Error setting expression '" + variable_list[i] +
                         " in domain " + std::to_string(domain_list[j]) +
                         "' in SINK. "
                         "Expression has already been defined.");

          if (var_split.size() > 0) {
            LU::FunctionVariableDefinition func_def2 = func_def;
            func_def2.m_fileVariable = var_split[i];
            function_var_map[key] = func_def2;
          } else {
            function_var_map[key] = func_def;
          }
        }
      }
      variable = variable->NextSiblingElement();
    }

    // Add function definition to map
    sinks.push_back(function_var_map);
    function = function->NextSiblingElement("SINK");
  }
}

void NESOReader::read_particle_species_initial(
    TiXmlElement *specie, std::pair<int, LU::FunctionVariableMap> &initial) {

  if (!specie) {
    return;
  }

  // Scan through conditions section looking for functions.
  TiXmlElement *function = specie->FirstChildElement("INITIAL");

  std::stringstream tagcontent;
  tagcontent << *function;

  // Every function must have a NAME attribute
  NESOASSERT(function->Attribute("N"),
             "INITIAL must have N attribute defined in XML "
             "element: \n\t'" +
                 tagcontent.str() + "'");
  std::string N_str = function->Attribute("N");
  NESOASSERT(!N_str.empty(), "Initial must have a non-empty N in XML "
                             "element: \n\t'" +
                                 tagcontent.str() + "'");

  int N = std::stoi(N_str);

  // Retrieve first entry (variable, or file)
  TiXmlElement *element = function;
  TiXmlElement *variable = element->FirstChildElement();

  // Create new function structure with default type of none.
  LU::FunctionVariableMap function_var_map;

  // Process all entries in the function block
  while (variable) {
    LU::FunctionVariableDefinition func_def;
    std::string condition_type = variable->Value();

    // If no var is specified, assume wildcard
    std::string variable_str;
    if (!variable->Attribute("VAR")) {
      variable_str = "*";
    } else {
      variable_str = variable->Attribute("VAR");
    }

    // Parse list of variables
    std::vector<std::string> variable_list;
    ParseUtils::GenerateVector(variable_str, variable_list);

    // If no domain is specified, put to 0
    std::string domain_str;
    if (!variable->Attribute("DOMAIN")) {
      domain_str = "0";
    } else {
      domain_str = variable->Attribute("DOMAIN");
    }

    // Parse list of domains
    std::vector<std::string> var_split;
    std::vector<unsigned int> domain_list;
    ParseUtils::GenerateSeqVector(domain_str, domain_list);

    // if no evars is specified, put "x y z t"
    std::string evars_str = "x y z t";
    if (variable->Attribute("EVARS")) {
      evars_str = evars_str + std::string(" ") + variable->Attribute("EVARS");
    }

    // Expressions are denoted by E
    if (condition_type == "E") {
      func_def.m_type = LU::eFunctionTypeExpression;

      // Expression must have a VALUE.
      NESOASSERT(variable->Attribute("VALUE"),
                 "Attribute VALUE expected for INITIAL.");
      std::string fcn_str = variable->Attribute("VALUE");
      NESOASSERT(!fcn_str.empty(),
                 (std::string("Expression for var: ") + variable_str +
                  std::string(" must be specified."))
                     .c_str());

      // set expression
      func_def.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
          this->interpreter, fcn_str, evars_str);
    }

    // Files are denoted by F
    else if (condition_type == "F") {
      // Check if transient or not
      if (variable->Attribute("TIMEDEPENDENT") &&
          boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
        func_def.m_type = LU::eFunctionTypeTransientFile;
      } else {
        func_def.m_type = LU::eFunctionTypeFile;
      }

      // File must have a FILE.
      NESOASSERT(variable->Attribute("FILE"),
                 "Attribute FILE expected for function INITIAL.");
      std::string filename_str = variable->Attribute("FILE");
      NESOASSERT(!filename_str.empty(),
                 "A filename must be specified for the FILE "
                 "attribute of function INITIAL.");

      std::vector<std::string> f_split;
      boost::split(f_split, filename_str, boost::is_any_of(":"));
      NESOASSERT(f_split.size() == 1 || f_split.size() == 2,
                 "Incorrect filename specification in INITIAL. "
                 "Specify variables inside file as: "
                 "filename:var1,var2");

      // set the filename
      fs::path fullpath = f_split[0];
      func_def.m_filename = fullpath.string();

      if (f_split.size() == 2) {
        NESOASSERT(variable_list[0] != "*",
                   "Filename variable mapping not valid "
                   "when using * as a variable inside "
                   "INITIAL");

        boost::split(var_split, f_split[1], boost::is_any_of(","));
        NESOASSERT(var_split.size() == variable_list.size(),
                   "Filename variables should contain the "
                   "same number of variables defined in "
                   "VAR in INITIAL.");
      }
    }

    // Nothing else supported so throw an error
    else {
      std::stringstream tagcontent;
      tagcontent << *variable;

      NESOASSERT(false, "Identifier " + condition_type + " in function " +
                            std::string(function->Attribute("NAME")) +
                            " is not recognised in XML element: \n\t'" +
                            tagcontent.str() + "'");
    }

    // Add variables to function
    for (unsigned int i = 0; i < variable_list.size(); ++i) {
      for (unsigned int j = 0; j < domain_list.size(); ++j) {
        // Check it has not already been defined
        std::pair<std::string, int> key(variable_list[i], domain_list[j]);
        auto fcns_iter = function_var_map.find(key);
        NESOASSERT(fcns_iter == function_var_map.end(),
                   "Error setting expression '" + variable_list[i] +
                       " in domain " + std::to_string(domain_list[j]) +
                       "' in INITIAL. "
                       "Expression has already been defined.");

        if (var_split.size() > 0) {
          LU::FunctionVariableDefinition func_def2 = func_def;
          func_def2.m_fileVariable = var_split[i];
          function_var_map[key] = func_def2;
        } else {
          function_var_map[key] = func_def;
        }
      }
    }
    variable = variable->NextSiblingElement();
  }

  // Add function definition to map
  initial = std::make_pair(N, function_var_map);
}

void NESOReader::read_particle_species_boundary(
    TiXmlElement *specie, ParticleSpeciesBoundaryList &boundary) {
  if (!specie) {
    return;
  }

  // Read REGION tags
  TiXmlElement *boundary_conditions_element =
      specie->FirstChildElement("BOUNDARYINTERACTION");

  if (boundary_conditions_element) {
    TiXmlElement *region_element =
        boundary_conditions_element->FirstChildElement("REGION");

    // Read C(Composite), P (Periodic) tags
    while (region_element) {

      int boundary_region_id;
      int err = region_element->QueryIntAttribute("REF", &boundary_region_id);
      NESOASSERT(err == TIXML_SUCCESS,
                 "Error reading boundary region reference.");

      NESOASSERT(boundary.count(boundary_region_id) == 0,
                 "Boundary region '" + std::to_string(boundary_region_id) +
                     "' appears multiple times.");

      // Find the boundary region corresponding to this ID.
      std::string boundary_region_id_str;
      std::ostringstream boundary_region_id_strm(boundary_region_id_str);
      boundary_region_id_strm << boundary_region_id;

      TiXmlElement *condition_element = region_element->FirstChildElement();

      while (condition_element) {
        // Check type.
        std::string condition_type = condition_element->Value();

        if (condition_type == "C") {

          // All species are reflect.
          boundary[boundary_region_id] =
              ParticleBoundaryConditionType::eReflective;
        }

        else if (condition_type == "P") {

          boundary[boundary_region_id] =
              ParticleBoundaryConditionType::ePeriodic;
        }
        condition_element = condition_element->NextSiblingElement();
      }
      region_element = region_element->NextSiblingElement("REGION");
    }
  }
}

void NESOReader::read_particle_species(TiXmlElement *particles) {
  TiXmlElement *species = particles->FirstChildElement("SPECIES");
  if (species) {
    TiXmlElement *specie = species->FirstChildElement("S");

    while (specie) {
      std::stringstream tagcontent;
      tagcontent << *specie;
      std::string id = specie->Attribute("ID");
      NESOASSERT(!id.empty(), "Missing ID attribute in Species XML "
                              "element: \n\t'" +
                                  tagcontent.str() + "'");

      std::string name = specie->Attribute("NAME");
      NESOASSERT(!name.empty(),
                 "NAME attribute must be non-empty in XML element:\n\t'" +
                     tagcontent.str() + "'");
      ParticleSpeciesMap species_map;
      std::get<0>(species_map) = name;

      TiXmlElement *parameter = specie->FirstChildElement("P");

      // Multiple nodes will only occur if there is a comment in
      // between definitions.
      while (parameter) {
        std::stringstream tagcontent;
        tagcontent << *parameter;
        TiXmlNode *node = parameter->FirstChild();

        while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
          node = node->NextSibling();
        }

        if (node) {
          // Format is "paramName = value"
          std::string line = node->ToText()->Value(), lhs, rhs;

          try {
            parse_equals(line, lhs, rhs);
          } catch (...) {
            NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                  "' in XML element: \n\t'" + tagcontent.str() +
                                  "'");
          }

          // We want the list of parameters to have their RHS
          // evaluated, so we use the expression evaluator to do
          // the dirty work.
          if (!lhs.empty() && !rhs.empty()) {
            NekDouble value = 0.0;
            try {
              LU::Equation expession(this->interpreter, rhs);
              value = expession.Evaluate();
            } catch (const std::runtime_error &) {
              NESOASSERT(false, "Error evaluating parameter expression"
                                " '" +
                                    rhs + "' in XML element: \n\t'" +
                                    tagcontent.str() + "'");
            }
            this->interpreter->SetParameter(lhs, value);
            boost::to_upper(lhs);
            std::get<1>(species_map)[lhs] = value;
          }
        }
        parameter = parameter->NextSiblingElement();
      }
      read_particle_species_initial(specie, std::get<2>(species_map));
      read_particle_species_sources(specie, std::get<3>(species_map));
      read_particle_species_sinks(specie, std::get<4>(species_map));
      read_particle_species_boundary(specie, std::get<5>(species_map));
      specie = specie->NextSiblingElement("S");

      this->particle_species[std::stoi(id)] = species_map;
    }
  }
}

void NESOReader::read_reactions(TiXmlElement *particles) {
  TiXmlElement *reactions_element = particles->FirstChildElement("REACTIONS");
  if (reactions_element) {
    TiXmlElement *reaction_r = reactions_element->FirstChildElement("R");

    while (reaction_r) {
      std::stringstream tagcontent;
      tagcontent << *reaction_r;
      std::string id = reaction_r->Attribute("ID");
      NESOASSERT(!id.empty(), "Missing ID attribute in Reaction XML "
                              "element: \n\t'" +
                                  tagcontent.str() + "'");
      std::string type = reaction_r->Attribute("TYPE");
      NESOASSERT(!type.empty(),
                 "TYPE attribute must be non-empty in XML element:\n\t'" +
                     tagcontent.str() + "'");
      ReactionMap reaction_map;
      std::get<0>(reaction_map) = type;
      std::string species = reaction_r->Attribute("SPECIES");
      std::vector<std::string> species_list;
      boost::split(species_list, species, boost::is_any_of(","));

      for (const auto &s : species_list) {
        NESOASSERT(this->species.find(std::stoi(s)) != this->species.end(),
                   "Species '" + s +
                       "' not found.  Ensure it is specified under the "
                       "<SPECIES> tag");
        std::get<1>(reaction_map).push_back(std::stoi(s));
      }

      TiXmlElement *rate = reaction_r->FirstChildElement("RATE");
      std::get<2>(reaction_map).first = rate->Attribute("TYPE");
      if (rate->Attribute("VALUE")) {
        std::get<2>(reaction_map).second = std::stod(rate->Attribute("VALUE"));
      }
      TiXmlElement *cross_section =
          reaction_r->FirstChildElement("CROSSSECTION");
      std::get<3>(reaction_map).first = cross_section->Attribute("TYPE");
      if (cross_section->Attribute("VALUE")) {
        std::get<3>(reaction_map).second = std::stod(rate->Attribute("VALUE"));
      }

      reaction_r = reaction_r->NextSiblingElement("R");
      this->reactions[std::stoi(id)] = reaction_map;
    }
  }
}

void NESOReader::read_particles() {
  // Check we actually have a document loaded.
  NESOASSERT(&this->session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&this->session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("NESO")
                  .FirstChildElement("PARTICLES")
                  .Element();

  if (!particles) {
    return;
  }
  read_parameters(particles);
  read_particle_species(particles);
  read_reactions(particles);
}

void NESOReader::load_particle_species_parameter(const int s,
                                                 const std::string &name,
                                                 int &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<1>(this->particle_species.at(s));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

void NESOReader::load_particle_species_parameter(const int s,
                                                 const std::string &name,
                                                 NekDouble &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<1>(this->particle_species.at(s));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  var = param_iter->second;
}

int NESOReader::get_particle_species_initial_N(const int s) const {
  return std::get<2>(this->particle_species.at(s)).first;
}
/**
 *
 */
LU::EquationSharedPtr NESOReader::get_particle_species_initial(
    const int s, const std::string &pVariable, const int pDomain) const {

  LU::FunctionVariableMap function =
      std::get<2>(this->particle_species.at(s)).second;

  // Check for specific and wildcard definitions
  std::pair<std::string, int> key(pVariable, pDomain);
  std::pair<std::string, int> defkey("*", pDomain);

  auto it2 = function.find(key);
  auto it3 = function.find(defkey);
  bool specific = it2 != function.end();
  bool wildcard = it3 != function.end();

  // Check function is defined somewhere
  ASSERTL0(specific || wildcard, "No such variable " + pVariable +
                                     " in domain " +
                                     boost::lexical_cast<std::string>(pDomain) +
                                     " defined for INITIAL in session file.");

  // If not specific, must be wildcard
  if (!specific) {
    it2 = it3;
  }

  ASSERTL0((it2->second.m_type == LU::eFunctionTypeExpression),
           std::string("Function is defined by a file."));
  return it2->second.m_expression;
}

/**
 *
 */
LU::EquationSharedPtr
NESOReader::get_particle_species_initial(const int s, const unsigned int &pVar,
                                         const int pDomain) const {
  ASSERTL0(pVar < this->session->GetVariables().size(),
           "Variable index out of range.");
  return get_particle_species_initial(s, this->session->GetVariables()[pVar],
                                      pDomain);
}

const std::vector<std::pair<int, LU::FunctionVariableMap>> &
NESOReader::get_particle_species_sources(const int s) const {
  return std::get<3>(this->particle_species.at(s));
}

const std::vector<LU::FunctionVariableMap> &
NESOReader::get_particle_species_sinks(const int s) const {
  return std::get<4>(this->particle_species.at(s));
}

const ParticleSpeciesBoundaryList &
NESOReader::get_particle_species_boundary(const int s) const {
  return std::get<5>(this->particle_species.at(s));
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, int &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, int &var,
                                const int &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    NekDouble param = round(param_iter->second);
    var = LU::checked_cast<int>(param);
  } else {
    var = pDefault;
  }
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, size_t &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, size_t &var,
                                const size_t &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    NekDouble param = round(param_iter->second);
    var = LU::checked_cast<int>(param);
  } else {
    var = pDefault;
  }
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, NekDouble &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  var = param_iter->second;
}

/**
 *
 */
void NESOReader::load_parameter(const std::string &name, NekDouble &var,
                                const NekDouble &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    var = param_iter->second;
  } else {
    var = pDefault;
  }
}
} // namespace NESO
