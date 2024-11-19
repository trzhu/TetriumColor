#include <vector>

#include "TetriumColor/PseudoIsochromaticPlateGenerator.h"

PseudoIsochromaticPlateGenerator::PseudoIsochromaticPlateGenerator(
    const std::vector<std::string>& transform_dirs,
    const std::vector<std::string>& pregenerated_filenames,
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

    printf("transform_dirs: ");
    for (const auto& dir : transform_dirs) {
        printf("%s ", dir.c_str());
    }
    printf("\n");

    // Convert vectors to Python lists
    PyObject* py_transform_dirs = PyList_New(transform_dirs.size());
    for (size_t i = 0; i < transform_dirs.size(); ++i) {
        PyList_SetItem(py_transform_dirs, i, PyUnicode_FromString(transform_dirs[i].c_str()));
    }

    PyObject* py_pregenerated_filenames = PyList_New(pregenerated_filenames.size());
    for (size_t i = 0; i < pregenerated_filenames.size(); ++i) {
        PyList_SetItem(py_pregenerated_filenames, i, PyUnicode_FromString(pregenerated_filenames[i].c_str()));
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
                py_transform_dirs,
                py_pregenerated_filenames,
                PyLong_FromLong(num_tests),
                PyLong_FromLong(seed)
            );
            pInstance = PyObject_CallObject(pClass, pArgs);
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
    Py_Finalize();
}

void PseudoIsochromaticPlateGenerator::NewPlate(
    const std::string& filename_RGB,
    const std::string& filename_OCV,
    int hidden_number
)
{
    if (pInstance != nullptr) {
        printf("iweofioaweio");
        PyObject* pValue = PyObject_CallMethod(
            pInstance, "NewPlate", "ssi", filename_RGB.c_str(), filename_OCV.c_str(), hidden_number
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
