#pragma once

#include <string>
#include "ColorTestResult.h"

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
        ColorTestResult result,
        const std::string& filename_RGB,
        const std::string& filename_OCV,
        int hidden_number
    );

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
};
} // namespace TetriumColor
