#pragma once

#include <Python.h>
#include <string>

namespace TetriumColor
{
class PseudoIsochromaticPlateGenerator
{
  public:
    PseudoIsochromaticPlateGenerator(
        const std::vector<std::string>& transform_dirs,
        const std::vector<std::string>& pregenerated_filenames,
        int num_tests,
        int seed = 42
    );

    ~PseudoIsochromaticPlateGenerator();

    void NewPlate(
        const std::string& filename_RGB,
        const std::string& filename_OCV,
        int hidden_number
    );

    void GetPlate(
        PyObject* previous_result,
        const std::string& filename_RGB,
        const std::string& filename_OCV,
        int hidden_number
    );

  private:
    PyObject* pModule;
    PyObject* pClass;
    PyObject* pInstance;
};
} // namespace TetriumColor
