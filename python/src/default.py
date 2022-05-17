
def default_parameters():
  params = {"input_scale":1,
            "analog_resample":1000,
            "IA_TF_file":'./src/AFE/AFE_data/IA_dist.dat',
            "IA_DCout": 0.6,
            "IA_thermal_noise" : 1e-07, #0.095e-6,
            "IA_flicker_noise_corner" : None, #1,
            "ADC_Fs": 200,
            "ADC_VCO_f0": 4550000, #26758898,
            "ADC_VCO_TF_file":'./src/AFE/AFE_data/VCO_TF.dat',
            "ADC_thermal_noise" : 0.01e-3,
            "beat_freq": 1.6,
            "SVM_library":'./c_lib_fast/',
            "SVM_model_file":'c_src/svm_model.h',
            "SVM_feature_select":[0,1,43,33,55,39,57,38,54,58,56,37,40,60,52,61,36,48,32,50], #[0,1,18,8,30,14,32,13,29,33,31,12,15,35,28,36,11,23,37,25]
            "SVM_feature_N_V":10,
            "SVM_feature_N_S":6,
            "SVM_SV_N_V":114,
            "SVM_SV_N_S":200,
            "SVM_pruning_D":0,
            "SVM_C":0.1,
            "SVM_gamma":0.1,
            "save_feature":False}
  return params