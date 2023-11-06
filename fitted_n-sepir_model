# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate, optimize


xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
ydatasuffolkold = [0, 0, 1, 0, 0, 1, 1, 3, 13, 14, 23, 55, 60, 55, 92, 210, 275, 399, 523, 535, 521, 355, 696, 834, 935, 1002, 1103, 760, 799, 1253, 1166, 1193, 1237, 1198, 1013, 918, 1484, 1436, 1423, 1097, 966, 797, 525, 643, 899, 1021, 1082, 781, 596, 376, 809, 686, 689, 536, 503, 302, 205, 463, 471, 432, 357, 381, 178, 124, 294, 360, 337, 251, 224, 125, 80, 215, 231, 203, 185, 172, 84, 54, 132, 121, 174, 225, 152, 66, 57, 48, 150, 132, 125, 99, 54, 29, 116, 84, 57, 56, 54, 24, 22, 62, 48, 41, 41, 48, 31, 21, 56, 66, 60, 44, 38, 39, 23, 53, 59, 59, 51, 52, 38, 30]
ydatanassauold = [0, 0, 0, 2, 1, 2, 7, 4, 18, 17, 24, 63, 87, 112, 183, 347, 493, 691, 802, 685, 527, 424, 768, 814, 977, 1185, 1086, 899, 838, 1382, 1142, 1197, 1244, 1252, 1028, 975, 1748, 1320, 1264, 1132, 1014, 829, 575, 638, 695, 744, 823, 642, 435, 346, 479, 427, 483, 507, 397, 232, 159, 373, 374, 309, 294, 263, 155, 89, 205, 211, 247, 176, 153, 77, 67, 157, 156, 132, 142, 110, 70, 30, 110, 129, 164, 149, 114, 46, 39, 62, 135, 101, 125, 99, 53, 30, 110, 69, 43, 65, 57, 28, 11, 59, 62, 42, 46, 34, 17, 19, 46, 44, 40, 34, 52, 31, 24, 45, 42, 52, 45, 52, 24, 300]
ydatanycold = [2, 0, 0, 5, 3, 6, 8, 22, 60, 70, 159, 370, 617, 640, 1052, 2108, 2432, 2977, 3695, 3986, 2638, 2588, 3586, 4481, 4887, 5070, 5143, 3501, 3548, 6152, 5519, 5511, 5830, 5721, 4019, 3833, 6480, 6165, 5637, 5095, 4660, 3848, 2825, 3355, 4217, 3974, 3648, 3680, 2270, 2428, 3892, 3179, 3585, 2908, 2683, 1623, 1066, 2377, 2870, 2487, 2111, 2001, 1154, 830, 1660, 1709, 1669, 1370, 1215, 710, 510, 1340, 1419, 1458, 1238, 976, 534, 398, 1000, 1094, 1153, 1146, 1134, 511, 495, 500, 1135, 815, 703, 695, 378, 239, 731, 609, 548, 570, 425, 259, 214, 481, 473, 406, 413, 435, 197, 181, 398, 453, 368, 402, 381, 216, 172, 410, 392, 400, 329, 336, 213, 206]
ydatasuffolk = [0, 0, 1, 1, 1, 2, 3]
ydatanassau = [0, 0, 0, 2, 3, 5, 12]
ydatanyc = [2, 0, 0, 7, 10, 16, 24]


xdata = np.array(xdata, dtype=np.float64)
ydatasuffolk = np.array(ydatasuffolkold, dtype=np.float64)
ydatanassau = np.array(ydatanassauold, dtype=np.float64)
ydatanyc = np.array(ydatanycold, dtype=np.float64)


newpop = [8470000*0.015, 1390000*0.015, 1530000*0.015]
N1 = newpop[0]
N2 = newpop[1]
N3 = newpop[2]

def sepir_model(y, x, beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,
                      gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                      alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3):
    S1 = y[0]
    E1 = y[1]
    P1 = y[2]
    C1 = y[3]
    I1 = y[4]
    R1 = y[5]
    S2 = y[6]
    E2 = y[7]
    P2 = y[8]
    C2 = y[9]
    I2 = y[10]
    R2 = y[11]
    S3 = y[12]
    E3 = y[13]
    P3 = y[14]
    C3 = y[15]
    I3 = y[16]
    R3 = y[17]
    
    sum_ISPS1 = S1 * (beta11*(C1+I1) + gamma11*P1 + beta12*(C2+I2) + gamma12*P2 + beta13*(C3+I3) + gamma13*P3)/(newpop[0]);
    dS1dt = -sum_ISPS1;
    dE1dt = sum_ISPS1 - alpha1 * E1;
    dP1dt = alpha1 * E1 - tau1 * P1;
    dC1dt = tau1*P1 - 1*C1;
    dI1dt = 1*C1 - delta1 * I1;
    dR1dt = delta1 * I1;
    
    sum_ISPS2 = S2 * (beta21*(C1+I1) + gamma21*P1 + beta22*(C2+I2) + gamma22*P2 + beta23*(C3+I3) + gamma23*P3)/(newpop[1]);
    dS2dt = -sum_ISPS2;
    dE2dt = sum_ISPS2 - alpha2 * E2;
    dP2dt = alpha2 * E2 - tau2 * P2;
    dC2dt = tau2*P2 - 1*C2
    dI2dt = 1*C2 - delta2 * I2;
    dR2dt = delta2 * I2;
    
    sum_ISPS3 = S3 * (beta31*(C1+I1) + gamma31*P1 + beta32*(C2+I2) + gamma32*P2 + beta33*(C3+I3) + gamma33*P3)/(newpop[2]);
    dS3dt = -sum_ISPS3;
    dE3dt = sum_ISPS3 - alpha3 * E3;
    dP3dt = alpha3 * E3 - tau3 * P3;
    dC3dt = tau3*P3 - 1*C3
    dI3dt = 1*C3 - delta3 * I3;
    dR3dt = delta3 * I3;
    
    return (dS1dt, dE1dt, dP1dt, dC1dt, dI1dt, dR1dt, dS2dt, dE2dt, dP2dt, dC2dt, dI2dt, dR2dt, dS3dt, dE3dt, dP3dt, dC3dt, dI3dt, dR3dt)

def solve_ode(x, beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,
             gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
             alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3):
    solution = integrate.odeint(sepir_model, ((N1-33), 30, 2, 1, 0, 0, (N2-6), 3, 2, 1, 0, 0, (N3-6), 3, 2, 1, 0, 0),
                            x, args=(beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33, 
                                     gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                                     alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3))
    return solution[:, [3, 9, 15]]

real_Is = np.stack((ydatanyc, ydatanassau, ydatasuffolk)).T

def residual(z):
    global xdata
    beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,  \
        gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33, \
        alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3 \
        = z
    all_Is = solve_ode(xdata, beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33, 
                    gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                    alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(0,len(xdata)):
        sum1 += (all_Is[i][0] - real_Is[i][0])*(all_Is[i][0] - real_Is[i][0])
        sum2 += (all_Is[i][1] - real_Is[i][1])*(all_Is[i][1] - real_Is[i][1])
        sum3 += (all_Is[i][2] - real_Is[i][2])*(all_Is[i][2] - real_Is[i][2])
    
    
    return math.sqrt(sum1 + sum2 + sum3)

result = optimize.minimize(residual,
                             (0.2, 0.02, 0.01, 0.03, 0.18, 0.01, 0.02, 0.02, 0.17,   0.36, 0.04, 0.01, 0.2, 0.35, 0.02, 0.02, 0.03, 0.35,   0.4, 0.4, 0.4, 0.33, 0.33, 0.33, 0.142, 0.142, 0.142),
                             bounds=[(0.1, 0.33), (0.0, 0.04), (0.0, 0.04), (0.0, 0.04), (0.0, 0.23), (0.0, 0.04), (0.0, 0.04),
                                     (0.0, 0.04), (0.0, 0.23), (0.2, 0.5), (0.0, 0.08), (0.0, 0.08), (0.0, 0.08), (0.2, 0.5),
                                     (0.0, 0.08), (0.0, 0.08), (0.0, 0.08), (0.2, 0.5), (0.35, 0.5), (0.35, 0.5), (0.35, 0.5),
                                     (0.33, 0.5), (0.33, 0.5), (0.33, 0.5), (0.15, 0.17), (0.15, 0.17), (0.15, 0.17)])



fitted = solve_ode(xdata, *result.x)

fitted_nyc = fitted[:,0]
fitted_nas = fitted[:,1]
fitted_suf = fitted[:,2]

fig, axes = plt.subplots(3)

plt.rcParams["figure.figsize"] = [12, 12]


axes[0].plot(xdata, ydatanyc, 'o')
axes[0].plot(xdata, fitted_nyc)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("NYC")
axes[1].plot(xdata, ydatanassau, 'o')
axes[1].plot(xdata, fitted_nas)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Nassau")
axes[2].plot(xdata, ydatasuffolk, 'o')
axes[2].plot(xdata, fitted_suf)
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Suffolk")
plt.show()
