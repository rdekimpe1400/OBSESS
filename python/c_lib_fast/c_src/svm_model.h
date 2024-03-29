//**********************************************
//
//  SVM model data
//
//**********************************************

#ifndef SVM_MOD_H_
#define SVM_MOD_H_

typedef int16_t feature_data_t;
typedef int32_t feature_dist_data_t;
typedef int32_t kernel_data_t;
typedef int32_t decision_data_t;
// Feature pre-processing
const int feature_select_idx_V[4] = {147, 149, 31, 119};
const feature_data_t scale_mean_V[4] = {-456, -117, -2228, 6};
const int32_t scale_std_V[4] = {22901, 74069, 35481, 201982};
const int feature_select_idx_S[4] = {0, 4, 5, 137};
const feature_data_t scale_mean_S[4] = {155, -1, -1, 0};
const int32_t scale_std_S[4] = {1596193, 2656765, 2489733, 656512};
const int scale_shift = 16;
// Exponential pre-calc coeff
const kernel_data_t exp_ai[21] = {65534, 65532, 65528, 65520, 65504, 65472, 65408, 65280, 65025, 64519, 63519, 61565, 57835, 51039, 39749, 24109, 8869, 1200, 21, 0, 0};
const int exp_ai_min = -15;
const int exp_ai_max = 5;
const int exp_shift = 16;
// SVM model data
const int n_feat_V = 4;
const int gam_inv_V = 40;
const int n_feat_S = 4;
const int gam_inv_S = 40;
const int feature_shift = 10;
const int feature_acc_shift = 2;
const int kernel_acc_shift = 16;
const int n_sv_class_V[2] = {23, 15};
const int n_sv_V = 38;
const int start_sv_V[2] = {0, 23};
const decision_data_t rho_V = 2232;
const decision_data_t sv_coef_V[38] = {-4670, -1661, -4825, -10896, -24517, -11892, -75724, -45224, -227274, -250316, -17097, -1097, -7997, -1385, -15274, -3926, -21, -53063, -1285, -3692, -1189, -7651, -475892, 51461, 1189, 1199, 18035, 44394, 5470, 415964, 5020, 4381, 19548, 7641, 25454, 60035, 419133, 167647};
const feature_data_t sv_V[38][4] = {
{-5710, -3250, 4558, -2175},
{1968, 3597, -5114, 4429},
{1796, 4084, -4979, 616},
{-662, -1335, -613, -899},
{779, 2160, -2378, 662},
{-1166, -770, 3454, -2006},
{-640, -573, -924, 521},
{-626, -1984, -1677, 567},
{134, 1130, -149, 650},
{-473, 204, -259, 413},
{-1293, -243, 2004, 348},
{-9946, -4698, 2461, -1627},
{6603, -2154, 4001, -3275},
{-1941, 839, 2352, 829},
{-5019, -533, 2614, 1097},
{773, 1840, 4322, -2480},
{11455, -2400, 4688, -2517},
{-1474, 1723, 2501, 314},
{8005, -1074, 4274, -4456},
{4627, 1475, 3634, -967},
{-9812, -1742, 1018, 1041},
{-5545, 983, 2223, 2136},
{-653, -677, -423, 302},
{-1318, -1419, 2029, -280},
{11254, -4847, 6304, -5954},
{-1276, -2045, 3157, 1248},
{-829, -281, 645, -2607},
{-633, -2379, -162, -9},
{-1062, -1258, -3155, 1350},
{-985, -1008, -718, 511},
{-444, -2357, 33, -1590},
{1625, 4182, 6491, 444},
{2304, -2208, 2127, -3328},
{1315, -2695, 4631, -64},
{331, 2003, 1208, -107},
{1824, 2543, -214, -181},
{-400, 1080, -194, 672},
{-936, -64, -325, 518}
};
const int n_sv_class_S[2] = {6, 7};
const int n_sv_S = 13;
const int start_sv_S[2] = {0, 6};
const decision_data_t rho_S = -5399;
const decision_data_t sv_coef_S[13] = {-28729, -16249, -198808, -1384, -322906, -16489, 773, 1909, 2590, 19498, 2611, 190507, 366676};
const feature_data_t sv_S[13][4] = {
{1686, 515, 3864, -87},
{907, -1187, 1470, 143},
{-1430, -254, -200, -97},
{-1479, -254, -48, -1178},
{-1381, -11, 141, 714},
{-797, -1471, -2860, -6},
{5559, 5704, 4510, -177},
{5827, 7082, -4455, -46},
{4122, 5866, 6371, 3},
{396, 2096, 4813, 83},
{-188, 961, 3028, -97},
{-1065, 28, 179, 253},
{-1114, -52, -200, 393}
};

#endif //SVM_MOD_H_
