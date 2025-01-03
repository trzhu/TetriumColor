#include <vector>

#include "TetriumColor/PseudoIsochromaticPlateGenerator.h"

namespace TetriumColor
{
PseudoIsochromaticPlateGenerator::PseudoIsochromaticPlateGenerator(
    const std::vector<std::string>& transform_dirs,
    const std::vector<std::string>& pregenerated_filenames,
    int num_tests,
    int seed
)
{
    // Convert vectors to Python lists
    PyObject* py_transform_dirs = PyList_New(transform_dirs.size());
    for (size_t i = 0; i < transform_dirs.size(); ++i) {
        PyList_SetItem(py_transform_dirs, i, PyUnicode_FromString(transform_dirs[i].c_str()));
    }

    PyObject* py_pregenerated_filenames = PyList_New(pregenerated_filenames.size());
    for (size_t i = 0; i < pregenerated_filenames.size(); ++i) {
        PyList_SetItem(
            py_pregenerated_filenames, i, PyUnicode_FromString(pregenerated_filenames[i].c_str())
        );
    }

    // Import the Python module
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.TetraPlate");
    pModule = reinterpret_cast<PyObject*>(PyImport_Import(pName));
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the Python class
        pClass = reinterpret_cast<PyObject*>(PyObject_GetAttrString(reinterpret_cast<PyObject*>(pModule), "PseudoIsochromaticPlateGenerator"));

        if (pClass && PyCallable_Check(reinterpret_cast<PyObject*>(pClass)) == 1) {
            // Create an instance of the Python class
            PyObject* pArgs = PyTuple_Pack(
                4,
                py_transform_dirs,
                py_pregenerated_filenames,
                PyLong_FromLong(num_tests),
                PyLong_FromLong(seed)
            );
            pInstance = reinterpret_cast<PyObject*>(PyObject_CallObject(
                reinterpret_cast<PyObject*>(pClass)
                , pArgs));
            Py_DECREF(pArgs);
        } else {
            PyErr_Print();
            exit(-1);
        }
    } else {
        PyErr_Print();
        exit(-1);
    }

    Py_DECREF(py_transform_dirs);
    Py_DECREF(py_pregenerated_filenames);
}

PseudoIsochromaticPlateGenerator::~PseudoIsochromaticPlateGenerator()
{
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
}

void PseudoIsochromaticPlateGenerator::NewPlate(
    const std::string& filename_RGB,
    const std::string& filename_OCV,
    int hidden_number
)
{
    if (pInstance != nullptr) {
        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance), "NewPlate", "ssi", filename_RGB.c_str(), filename_OCV.c_str(), hidden_number
        );
        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        printf("gg, exiting\n");
        exit(-1);
    }
}

void PseudoIsochromaticPlateGenerator::GetPlate(
    ColorTestResult result,
    const std::string& filename_RGB,
    const std::string& filename_OCV,
    int hidden_number
)
{
    PyObject* pResult = PyLong_FromLong(static_cast<int>(result));
    if (pInstance != nullptr) {
        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance),
            "GetPlate",
            "Ossi",
            pResult,
            filename_RGB.c_str(),
            filename_OCV.c_str(),
            hidden_number
        );
        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    }
}

} // namespace TetriumColor
