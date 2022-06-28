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
const int feature_select_idx_V[10] = {147, 149, 31, 119, 145, 34, 152, 123, 160, 158};
const feature_data_t scale_mean_V[10] = {-392, -94, -2199, 10, 9, 39, 4, -24, -23, 1};
const int32_t scale_std_V[10] = {24818, 72955, 33863, 87229, 90029, 118674, 116811, 168838, 44138, 347356};
const int feature_select_idx_S[6] = {0, 4, 5, 137, 1, 161};
const feature_data_t scale_mean_S[6] = {155, -1, -1, -1, 155, 14};
const int32_t scale_std_S[6] = {1593834, 2659711, 2492391, 344920, 1592497, 50559};
const int scale_shift = 16;
// Exponential pre-calc coeff
const kernel_data_t exp_ai[21] = {65534, 65532, 65528, 65520, 65504, 65472, 65408, 65280, 65025, 64519, 63519, 61565, 57835, 51039, 39749, 24109, 8869, 1200, 21, 0, 0};
const int exp_ai_min = -15;
const int exp_ai_max = 5;
const int exp_shift = 16;
// SVM model data
const int n_feat_V = 10;
const int gam_inv_V = 100;
const int n_feat_S = 6;
const int gam_inv_S = 60;
const int feature_shift = 10;
const int feature_acc_shift = 3;
const int kernel_acc_shift = 16;
const int n_sv_class_V[2] = {13, 12};
const int n_sv_V = 25;
const int start_sv_V[2] = {0, 13};
const decision_data_t rho_V = 2935;
const decision_data_t sv_coef_V[25] = {-2575, -16157, -2604, -1734, -930, -186, -54, -393, -722, -9322, -1264, -1479, -3915, 3822, 365, 3933, 4300, 2258, 684, 10373, 810, 504, 670, 10810, 2806};
const feature_data_t sv_V[25][10] = {
{582, -1106, -1650, -115, 485, -1214, 490, 352, 192, -42},
{3, -426, -2033, 19, -228, 554, -1051, -768, -101, 25},
{53, -445, 14, 272, -671, 3075, -1703, 230, -70, -122},
{125, -258, 20, 411, 539, 837, 722, -44, 102, -5},
{552, 342, 200, 40, -311, -158, 1399, -258, 274, 73},
{3448, -1709, 103, -104, -208, -3842, 142, -549, -1747, -657},
{517, 624, 2151, 525, 555, 927, -584, -330, 630, -106},
{770, -201, 2933, 123, -2380, -218, 779, 1717, 1265, 243},
{1691, -432, 3904, -469, -1312, 1719, -2320, 808, 505, -387},
{626, -917, 2852, 314, 221, -88, 529, 648, 832, -5},
{1072, -288, 3168, -124, 151, 619, 2376, 426, 841, 105},
{350, -307, 64, 41, -101, 1295, 1351, -163, -522, -21},
{293, -37, -1008, -189, -759, 1668, 1554, 630, 1243, 41},
{-1778, -876, -22, 143, -199, -1656, -999, -284, -742, -32},
{-1953, -406, -999, 651, -826, -1830, 435, 439, 503, -11},
{-2099, -979, -818, 272, -138, -1009, -808, -714, -430, -5},
{-1788, -1309, -258, -112, -517, -1516, -218, -593, -199, -64},
{663, -2462, 1710, -1029, 1322, 4734, 627, -1544, 149, -95},
{-2136, -2935, -871, 229, 405, 489, -63, -513, -155, 502},
{-131, -2503, 1756, -1178, 1010, 2291, -208, -1484, 1902, 100},
{3955, -900, -5, 220, 4760, 6425, 629, -325, 2880, 608},
{-63, -54, -142, -133, -1112, -2192, -2366, 2665, -4014, 423},
{621, 1784, 650, 187, -513, -2460, -794, 1449, -4501, 455},
{-319, 3299, -2356, 734, -1047, 310, 1005, 1841, -163, -133},
{-1017, -2629, 1403, -1182, -1623, -146, 87, -1180, 3, -21}
};
const int n_sv_class_S[2] = {6, 6};
const int n_sv_S = 12;
const int start_sv_S[2] = {0, 6};
const decision_data_t rho_S = -1189;
const decision_data_t sv_coef_S[12] = {-7210, -272, -979, -3706, -208, -13499, 314, 431, 8394, 2150, 3760, 10825};
const feature_data_t sv_S[12][6] = {
{-553, -1471, 1207, 178, 1099, -229},
{-1015, -1309, 1131, 36, 491, -2096},
{-358, -3054, -4649, -184, -1500, -1267},
{-1964, -741, -808, -1495, -2035, 281},
{-1964, -1106, -884, -2663, -1865, -1877},
{-2012, -254, -237, 688, -2010, 687},
{5623, 6401, 2348, -58, 3286, 100},
{1416, 4129, 864, 715, -504, -7586},
{-577, 314, -542, -110, -1111, -37},
{-650, 720, 408, 183, -820, 84},
{-529, 232, -656, 78, -1087, -233},
{-480, 354, -352, 152, -917, 522}
};

#endif //SVM_MOD_H_
