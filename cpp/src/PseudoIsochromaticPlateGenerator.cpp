#include "TetriumColor/PseudoIsochromaticPlateGenerator.h"


PseudoIsochromaticPlateGenerator::PseudoIsochromaticPlateGenerator(
    const std::string& transform_dirs,
    const std::string& pregenerated_filenames,
    int num_tests,
    int seed
)
{
    Py_Initialize();

    { // configure the Python path
        PyObject* sys_path = PyImport_ImportModule("sys");
        if (sys_path != nullptr) {
            PyObject* sys_path_obj = PyObject_GetAttrString(sys_path, "path");
            if (sys_path_obj != nullptr) {
                // Append the directory to the sys.path
                PyList_Append(sys_path_obj, PyUnicode_FromString(TETRIUM_COLOR_MODULE_PATH));
            }
        }
    }

    // Import the Python module
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.TetraPlate");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the Python class
        pClass = PyObject_GetAttrString(pModule, "PseudoIsochromaticPlateGenerator");

        if (pClass && PyCallable_Check(pClass)) {
            // Create an instance of the Python class
            PyObject* pArgs = PyTuple_Pack(
                4,
                PyUnicode_FromString(transform_dirs.c_str()),
                PyUnicode_FromString(pregenerated_filenames.c_str()),
                PyLong_FromLong(num_tests),
                PyLong_FromLong(seed)
            );
            pInstance = PyObject_CallObject(pClass, pArgs);
            Py_DECREF(pArgs);
        } else {
            PyErr_Print();
        }
    } else {
        PyErr_Print();
    }
}

PseudoIsochromaticPlateGenerator::~PseudoIsochromaticPlateGenerator()
{
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
    Py_Finalize();
}

void PseudoIsochromaticPlateGenerator::NewPlate(
    const std::string& filename_RGB,
    const std::string& filename_OCV,
    int hidden_number
)
{
    if (pInstance != nullptr) {
        PyObject* pValue = PyObject_CallMethod(
            pInstance, "NewPlate", "ssi", filename_RGB.c_str(), filename_OCV.c_str(), hidden_number
        );
        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    }
}

void PseudoIsochromaticPlateGenerator::GetPlate(
    PyObject* previous_result,
    const std::string& filename_RGB,
    const std::string& filename_OCV,
    int hidden_number
)
{
    if (pInstance != nullptr) {
        PyObject* pValue = PyObject_CallMethod(
            pInstance,
            "GetPlate",
            "Ossi",
            previous_result,
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
