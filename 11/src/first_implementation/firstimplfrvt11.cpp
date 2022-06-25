/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */


#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "firstimplfrvt11.h"
#include <filesystem>


// #define DEBUG
#ifdef DEBUG
#define LOGD(m) cout<< m << endl
#else
#define LOGD(m) 
#endif
using namespace std;
using namespace FRVT;
using namespace FRVT_11;



FirstImplFRVT11::FirstImplFRVT11() {}

FirstImplFRVT11::~FirstImplFRVT11() {}

ReturnStatus
FirstImplFRVT11::initialize(const std::string &configDir)
{
    this->configDir = configDir;

    LOGD("[+] current working dir :>>>> "); 
    LOGD(filesystem::current_path().string().c_str());    

    LOGD("[+] setting PYTHONHOME and PYTHONPATH :>>>> "); 

    std::string python_path = filesystem::current_path().string() + "/" + this->configDir + "/python3_9_13";

    std::string env_python_path = "PYTHONPATH";
    setenv(env_python_path.c_str(), python_path.c_str(), 1);
    LOGD("PYTHONPATH: ");
    LOGD(std::getenv("PYTHONPATH"));

    std::string env_python_home = "PYTHONHOME";
    setenv(env_python_home.c_str(), python_path.c_str(), 1);
    LOGD("PYTHONHOME: ");
    LOGD(std::getenv("PYTHONHOME"));
    
    // set LD_LIBRARY_PATH
    string LD_LIBRARY_PATH = "LD_LIBRARY_PATH";
    const char* env_ld_path = getenv(LD_LIBRARY_PATH.c_str());
    string env_ld_path_str = env_ld_path;
    string UPDATED_LD_LIBRARY_PATH = env_ld_path_str + ":" + python_path + "/lib";
    setenv(LD_LIBRARY_PATH.c_str(), UPDATED_LD_LIBRARY_PATH.c_str(), 1);    




    
    LOGD("[+] Initializing python"); 
    Py_Initialize();
    LOGD("[+] Python Initialized");


    LOGD("Add config folder to sys path  \n");

    std::string command_sys_path_to_append = "import sys, os\n"
                    "sys.path.append(os.path.join(os.getcwd(), '" + this->configDir + "'))\n";
    LOGD("sys path to append");
    LOGD(command_sys_path_to_append);
    PyRun_SimpleString(command_sys_path_to_append.c_str());

#ifdef DEBUG
    PyRun_SimpleString("import os\n"
                    "print('sys.path = ', sys.path)\n");

   

    PyRun_SimpleString("import fr\n"
                    "print(fr)");


#endif


    LOGD("Importing python file  \n");
    // import module ...
    PyObject *p_module_name = PyUnicode_DecodeFSDefault("fr"); // fr.py file
    this->p_module = PyImport_Import(p_module_name);
    if (p_module == NULL)
    {
        LOGD("Cannot import python module !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    // Py_IncRef(p_module);

    LOGD("Importing 2 functions for createTemplate \n");

    this->p_function_multiple_images_single_face = PyObject_GetAttrString(p_module, "create_template_multiple_images_single_face");
    if (!(p_function_multiple_images_single_face && PyCallable_Check(p_function_multiple_images_single_face)))
    {
        LOGD("Cannot import  create_template_multiple_images_single_face !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("imported function :  create_template_multiple_images_single_face !  \n");
    // Py_IncRef(p_function_multiple_images_single_face);

    this->p_function_single_images_mutiple_faces = PyObject_GetAttrString(p_module, "create_template_single_images_mutiple_faces");
    if (!(p_function_single_images_mutiple_faces && PyCallable_Check(p_function_single_images_mutiple_faces)))
    {
        LOGD("Cannot import  create_template_single_images_mutiple_faces !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("imported function :  create_template_single_images_mutiple_faces !  \n");
    // Py_IncRef(p_function_single_images_mutiple_faces);
    
    LOGD("loading detection model ... \n");
    PyObject *outer_tuple_1 = PyTuple_New(1);
    PyTuple_SetItem(outer_tuple_1, 0, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    PyObject *p_fucntion_detection_model = PyObject_GetAttrString(p_module, "getDetectorModel");
    this->p_detection_model_object = PyObject_CallObject(p_fucntion_detection_model, outer_tuple_1);
    if (this->p_detection_model_object == NULL){
        LOGD("[+] detection model came NULL");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("detection model loaded ... \n");
    // Py_IncRef(p_detection_model_object);
    
    // LOGD("loading embedding model ... \n");
    // PyObject *outer_tuple_2 = PyTuple_New(1);
    // PyTuple_SetItem(outer_tuple_2, 0, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    // PyObject *p_fucntion_embedding_model = PyObject_GetAttrString(p_module, "getEmbeddingModel");
    // this->p_embedding_model_object = PyObject_CallObject(p_fucntion_embedding_model, outer_tuple_2);
    // if (this->p_embedding_model_object == NULL){
    //     LOGD("[+] embedding model came NULL");
    //     return ReturnStatus(ReturnCode::ConfigError);
    // }
    // LOGD("embedding model loaded ... \n");
    // // Py_IncRef(p_embedding_model_object);



    LOGD("loading FaceMesh ... \n");
    PyObject *p_fucntion_face_mesh_images = PyObject_GetAttrString(p_module, "getFaceMesh");
    this->p_face_mesh_images = PyObject_CallObject(p_fucntion_face_mesh_images, NULL);
    if (this->p_face_mesh_images == NULL){
        LOGD("[+] FaceMesh model came NULL");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("FaceMesh loaded ... \n");
    
    LOGD("initialized successfully ! \n");
    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
FirstImplFRVT11::createTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{

    // Py_IncRef(p_embedding_model_object);
    // Py_IncRef(p_detection_model_object);
    // Py_IncRef(p_function_single_images_mutiple_faces);
    // Py_IncRef(p_function_multiple_images_single_face);
    // Py_IncRef(p_module);

    this->p_function_multiple_images_single_face = PyObject_GetAttrString(p_module, "create_template_multiple_images_single_face");
    if (!(p_function_multiple_images_single_face && PyCallable_Check(p_function_multiple_images_single_face)))
    {
        LOGD("[+1] createTemplate Cannot import  create_template_multiple_images_single_face !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("[+1] createTemplate imported function :  create_template_multiple_images_single_face !  \n");
    // Py_IncRef(p_function_multiple_images_single_face);

    this->p_function_single_images_mutiple_faces = PyObject_GetAttrString(p_module, "create_template_single_images_mutiple_faces");
    if (!(p_function_single_images_mutiple_faces && PyCallable_Check(p_function_single_images_mutiple_faces)))
    {
        LOGD("[+1] createTemplate Cannot import  create_template_single_images_mutiple_faces !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("[+1] createTemplate imported function :  create_template_single_images_mutiple_faces !  \n");
    // Py_IncRef(p_function_single_images_mutiple_faces);
    

    // LOGD("loading detection model ... \n");
    // PyObject *outer_tuple_1 = PyTuple_New(1);
    // PyTuple_SetItem(outer_tuple_1, 0, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    // PyObject *p_fucntion_detection_model = PyObject_GetAttrString(p_module, "getDetectorModel");
    // this->p_detection_model_object = PyObject_CallObject(p_fucntion_detection_model, outer_tuple_1);
    // if (this->p_detection_model_object == NULL){
    //     LOGD("[+1] detection model came NULL");
    //     return ReturnStatus(ReturnCode::ConfigError);
    // }
    // LOGD("detection model loaded ... \n");
    // Py_IncRef(p_detection_model_object);
    
    // Py_DecRef(p_embedding_model_object);
    // PyMem_Free(p_embedding_model_object);
    // // PyMem_RawFree(p_embedding_model_object);
    // p_embedding_model_object = NULL;

    // LOGD("loading embedding model ... \n");
    // PyObject *outer_tuple_2 = PyTuple_New(1);
    // PyTuple_SetItem(outer_tuple_2, 0, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    // PyObject *p_fucntion_embedding_model = PyObject_GetAttrString(p_module, "getEmbeddingModel");
    // this->p_embedding_model_object = PyObject_CallObject(p_fucntion_embedding_model, outer_tuple_2);
    // if (this->p_embedding_model_object == NULL){
    //     LOGD("[+1] embedding model came NULL");
    //     return ReturnStatus(ReturnCode::ConfigError);
    // }
    // LOGD("embedding model loaded ... \n");
    // Py_IncRef(p_embedding_model_object);


    // 1. Multiface
    // ( first-face=> ((meta), (data)), ... )
    LOGD("[+1] building args ... ");
    // tuple of length N-images of same person
    PyObject *p_total_faces_tuple = PyTuple_New(faces.size());
    LOGD("[+1] face for loop begins with length ");
    LOGD(faces.size());
    for (int face_i=0; face_i<faces.size(); face_i++)
    {
        
        PyObject *p_single_face_tuple = PyTuple_New(2); // meta and data
        // fill meta
        PyObject *p_single_face_metadata = PyTuple_New(4); // meta 
        PyTuple_SetItem(p_single_face_metadata, 0, PyLong_FromLong((int)faces[face_i].description));
        PyTuple_SetItem(p_single_face_metadata, 1, PyLong_FromLong(faces[face_i].width));
        PyTuple_SetItem(p_single_face_metadata, 2, PyLong_FromLong(faces[face_i].height));
        PyTuple_SetItem(p_single_face_metadata, 3, PyLong_FromLong(faces[face_i].depth));
        
        long length_of_data=0;
        if (faces[face_i].depth == 8)
        {
            length_of_data = faces[face_i].width * faces[face_i].height;
        }else
        {
            length_of_data = faces[face_i].width * faces[face_i].height * 3;
        }

        PyObject *p_single_face_data = PyTuple_New(length_of_data); // imagedata
        
        for (int data_i=0; data_i<length_of_data; data_i++)
        {
            PyTuple_SetItem(p_single_face_data, data_i, PyLong_FromLong(faces[face_i].data.get()[data_i]));
        }
            
        PyTuple_SetItem(p_single_face_tuple, 0, p_single_face_metadata);
        PyTuple_SetItem(p_single_face_tuple, 1, p_single_face_data);

        PyTuple_SetItem(p_total_faces_tuple, face_i, p_single_face_tuple);
    }
    LOGD("[+1] face for loop ends ... ");
    PyObject *full_args = PyTuple_New(5); // template role, face tuple, 2 models
    PyTuple_SetItem(full_args, 0, PyLong_FromLong((int)role));
    PyTuple_SetItem(full_args, 1, p_total_faces_tuple);
    PyTuple_SetItem(full_args, 2, this->p_detection_model_object);
    PyTuple_SetItem(full_args, 3, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    PyTuple_SetItem(full_args, 4, this->p_face_mesh_images);
    LOGD("[+1] full args ready ... ");


    PyObject *outer_tuple = PyTuple_New(1);
    PyTuple_SetItem(outer_tuple, 0, full_args);
    LOGD("[+1] calling function create_template_multiple_images_single_face");
    PyObject *p_return_tuple = PyObject_CallObject(p_function_multiple_images_single_face, outer_tuple);
    LOGD("[+1] return tuple address :: ");
    LOGD(p_return_tuple);
    if (p_return_tuple != NULL)
    {
        LOGD("[+1] called python function successfully : ");
        LOGD(p_return_tuple);
        LOGD(PyTuple_Size(p_return_tuple));
        PyObject *p_return_code = PyTuple_GetItem(p_return_tuple, 0);
        LOGD("[+1] got return code: ");
        PyObject *p_template_tuple = PyTuple_GetItem(p_return_tuple, 1);
        LOGD("[+1] got template tuple : ");
        PyObject *p_eye_details_tuple = PyTuple_GetItem(p_return_tuple, 2);

        LOGD("[+1] filling template : ");
        // fill template details
        std::vector<float> fv;
        for (int tmp_i=0; tmp_i<this->featureVectorSize; tmp_i++)
        {
            fv.push_back(PyLong_AsDouble(PyTuple_GetItem(p_template_tuple, tmp_i)));
        }

        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
        int dataSize = sizeof(float) * fv.size();
        templ.resize(dataSize);
        memcpy(templ.data(), bytes, dataSize);

        LOGD("[+1] filling eye details: ");
        for (unsigned int i = 0; i < faces.size(); i++) 
        {
            PyObject *p_eye_info_tuple = PyTuple_GetItem(p_eye_details_tuple, i);
            eyeCoordinates.push_back(EyePair(PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 0)),
                                            PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 1)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 2)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 3)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 4)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 5))
                                    )
                            );
        
        } 
        LOGD("[+1] fucntion returned code : ");
        LOGD(PyLong_AsLong(p_return_code));
        if((int)PyLong_AsLong(p_return_code)==0)
        {
            return ReturnStatus(ReturnCode::Success);    
        }
        else
        {
            return ReturnStatus(ReturnCode::UnknownError); 
        }    
    }
    else
    {
        LOGD("[+1] error 175 ");
        return ReturnStatus(ReturnCode::UnknownError); 
    }
}

// #define LOGD(m) cout<< m << endl

ReturnStatus
FirstImplFRVT11::createTemplate(
    const FRVT::Image &image,
    FRVT::TemplateRole role,
    std::vector<std::vector<uint8_t>> &templs,
    std::vector<FRVT::EyePair> &eyeCoordinates)
{

        this->p_function_multiple_images_single_face = PyObject_GetAttrString(p_module, "create_template_multiple_images_single_face");
    if (!(p_function_multiple_images_single_face && PyCallable_Check(p_function_multiple_images_single_face)))
    {
        LOGD("[+2] createTemplate Cannot import  create_template_multiple_images_single_face !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("[+2] createTemplate imported function :  create_template_multiple_images_single_face !  \n");
    // Py_IncRef(p_function_multiple_images_single_face);

    this->p_function_single_images_mutiple_faces = PyObject_GetAttrString(p_module, "create_template_single_images_mutiple_faces");
    if (!(p_function_single_images_mutiple_faces && PyCallable_Check(p_function_single_images_mutiple_faces)))
    {
        LOGD("[+2] createTemplate Cannot import  create_template_single_images_mutiple_faces !  \n");
        return ReturnStatus(ReturnCode::ConfigError);
    }
    LOGD("[+2] createTemplate imported function :  create_template_single_images_mutiple_faces !  \n");
    // Py_IncRef(p_function_single_images_mutiple_faces);


    // 1. Multiface
    // ( first-face=> ((meta), (data)), ... )
    LOGD("[++] building args ... ");
    // tuple of length N-images of same person
    PyObject *p_total_faces_tuple = PyTuple_New(1);
    LOGD("[++] face for loop begins ... ");
    for (int face_i=0; face_i<1; face_i++)
    {
        
        PyObject *p_single_face_tuple = PyTuple_New(2); // meta and data
        // fill meta
        PyObject *p_single_face_metadata = PyTuple_New(4); // meta 
        PyTuple_SetItem(p_single_face_metadata, 0, PyLong_FromLong((int)image.description));
        PyTuple_SetItem(p_single_face_metadata, 1, PyLong_FromLong(image.width));
        PyTuple_SetItem(p_single_face_metadata, 2, PyLong_FromLong(image.height));
        PyTuple_SetItem(p_single_face_metadata, 3, PyLong_FromLong(image.depth));
        
        long length_of_data=0;
        if (image.depth == 8)
        {
            length_of_data = image.width * image.height;
        }else
        {
            length_of_data = image.width * image.height * 3;
        }

        PyObject *p_single_face_data = PyTuple_New(length_of_data); // imagedata
        
        for (int data_i=0; data_i<length_of_data; data_i++)
        {
            PyTuple_SetItem(p_single_face_data, data_i, PyLong_FromLong(image.data.get()[data_i]));
        }
            
        PyTuple_SetItem(p_single_face_tuple, 0, p_single_face_metadata);
        PyTuple_SetItem(p_single_face_tuple, 1, p_single_face_data);

        PyTuple_SetItem(p_total_faces_tuple, face_i, p_single_face_tuple);
    }
    LOGD("[++] face for loop ends ... ");
    PyObject *full_args = PyTuple_New(5); // template role, face tuple, 2 models
    PyTuple_SetItem(full_args, 0, PyLong_FromLong((int)role));
    PyTuple_SetItem(full_args, 1, p_total_faces_tuple);
    PyTuple_SetItem(full_args, 2, this->p_detection_model_object);
    PyTuple_SetItem(full_args, 3, PyUnicode_DecodeFSDefault(this->configDir.c_str()));
    PyTuple_SetItem(full_args, 4, this->p_face_mesh_images);
    LOGD("[++] full args ready ... ");
    

    PyObject *outer_tuple = PyTuple_New(1);
    PyTuple_SetItem(outer_tuple, 0, full_args);
    LOGD("[+2] calling function create_template_single_images_mutiple_faces");
    PyObject *p_return_tuple = PyObject_CallObject(p_function_single_images_mutiple_faces, outer_tuple);
    if (p_return_tuple != NULL)
    {
        LOGD("[++] called python function successfully : ");
        LOGD(p_return_tuple);   

        PyObject *p_return_code = PyTuple_GetItem(p_return_tuple, 0);
        LOGD("[++] got return code: ");
        LOGD(PyLong_AsLong(p_return_code));
        PyObject *p_multiple_templates_tuple = PyTuple_GetItem(p_return_tuple, 1);
        LOGD("[++] got template tuple ");
        PyObject *p_eye_details_tuple = PyTuple_GetItem(p_return_tuple, 2);
        int p_num_face_detected = PyTuple_Size(p_multiple_templates_tuple);

        LOGD("[++] got these many faces ");
        LOGD(p_num_face_detected);

        LOGD("[++] filling templates ");
        for (int i = 0; i < p_num_face_detected; i++) {
            std::vector<uint8_t> templ;
            std::vector<float> fv;
            LOGD("[++] getting ith template ");
            LOGD(i);
            PyObject *p_single_template_tuple = PyTuple_GetItem(p_multiple_templates_tuple, i);
            for (int tmp_i=0; tmp_i<this->featureVectorSize; tmp_i++)
            {
                fv.push_back(PyLong_AsDouble(PyTuple_GetItem(p_single_template_tuple, tmp_i)));
            }
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
            int dataSize = sizeof(float) * fv.size();
            templ.resize(dataSize);
            memcpy(templ.data(), bytes, dataSize);
            templs.push_back(templ);
        } 
        LOGD("[+2] filling eye details: ");
        for (unsigned int i = 0; i < p_num_face_detected; i++) 
        {
            PyObject *p_eye_info_tuple = PyTuple_GetItem(p_eye_details_tuple, i);
            eyeCoordinates.push_back(EyePair(PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 0)),
                                            PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 1)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 2)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 3)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 4)),
                                            (uint16_t)PyLong_AsLong(PyTuple_GetItem(p_eye_info_tuple, 5))
                                    )
                            );
        }
        LOGD("[++] fucntion returned code : ");
        LOGD(PyLong_AsLong(p_return_code));
        if((int)PyLong_AsLong(p_return_code)==0)
        {
            return ReturnStatus(ReturnCode::Success);    
        }
        else
        {
            return ReturnStatus(ReturnCode::UnknownError); 
        }
    }
    else
    {
    LOGD("[++] error, python function returned NULL");
    return ReturnStatus(ReturnCode::UnknownError);    
    }
}

ReturnStatus
FirstImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    LOGD("[+] 0.1 begins matching ... ");
    float *featureVectorEnroll = (float *)enrollTemplate.data();
    LOGD("[+] 0.2 ");
    float *featureVectorVerify = (float *)enrollTemplate.data();
    LOGD("[+] 0.3 ");
    double result = 0; 
    LOGD("[+] 0.4 for loop starts");
    for (unsigned int i=0; i<this->featureVectorSize; i++) {
        LOGD("[+] 0.4.1 ");
        LOGD(featureVectorEnroll[i]);
        LOGD("[+] 0.4.2 ");
        LOGD(featureVectorVerify[i]);
        float diff = featureVectorEnroll[i] - featureVectorVerify[i];
        LOGD("[+] 0.4.3 ");
        LOGD(diff);
        double diff_squared = (double)diff * (double)diff;
        LOGD("[+] 0.4.4 ");
        LOGD(diff_squared);
        result = result + diff_squared;
    } 

    LOGD("[+] 0.5 ");
    LOGD(result);
    similarity = result;
    LOGD("[+] 0.6 ");
    LOGD(similarity);
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<FirstImplFRVT11>();
}





