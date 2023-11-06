import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate, optimize


xdata_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

#newpop = [8470000, 1390000, 1530000]
newpop = [8470000*0.015, 1390000*0.015, 1530000*0.015]
N1 = newpop[0]
N2 = newpop[1]
N3 = newpop[2]

ydatasuffolk_all_raw = [1, 0, 0, 1, 1, 3, 13, 14, 23, 55, 60, 55, 92, 210, 275, 399, 523, 535, 521, 355, 696, 834, 935, 1002, 1103, 760, 799, 1253, 1166, 1193, 1237, 1198, 1013, 918, 1484, 1436, 1423, 1097, 966, 797, 525, 643, 899, 1021, 1082, 781, 596, 376, 809, 686, 689, 536, 503, 302, 205, 463, 471, 432, 357, 381, 178, 124, 294, 360, 337, 251, 224, 125, 80, 215, 231, 203, 185, 172, 84, 54, 132, 121, 174, 225, 152, 66, 57, 48, 150, 132, 125, 99, 54, 29]
ydatanassau_all_raw = [0, 2, 1, 2, 7, 4, 18, 17, 24, 63, 87, 112, 183, 347, 493, 691, 802, 685, 527, 424, 768, 814, 977, 1185, 1086, 899, 838, 1382, 1142, 1197, 1244, 1252, 1028, 975, 1748, 1320, 1264, 1132, 1014, 829, 575, 638, 695, 744, 823, 642, 435, 346, 479, 427, 483, 507, 397, 232, 159, 373, 374, 309, 294, 263, 155, 89, 205, 211, 247, 176, 153, 77, 67, 157, 156, 132, 142, 110, 70, 30, 110, 129, 164, 149, 114, 46, 39, 62, 135, 101, 125, 99, 53, 30]
ydatanyc_all_raw = [0, 5, 3, 6, 8, 22, 60, 70, 159, 370, 617, 641, 1051, 2111, 2432, 2976, 3694, 3986, 2637, 2588, 3585, 4482, 4884, 5068, 5136, 3499, 3548, 6150, 5518, 5510, 5828, 5719, 4019, 3831, 6481, 6163, 5638, 5096, 4659, 3848, 2825, 3355, 4216, 3973, 3648, 3678, 2270, 2428, 3891, 3179, 3585, 2908, 2683, 1623, 1066, 2377, 2870, 2487, 2111, 2001, 1154, 830, 1660, 1709, 1669, 1370, 1215, 710, 510, 1340, 1419, 1458, 1238, 976, 534, 398, 1000, 1094, 1153, 1146, 1133, 511, 496, 500, 1136, 815, 703, 695, 378, 239]

def smooth_data(ydata, window):
   ammended_ydata = [ydata[0]]*window + ydata
   smoothed_ydata = []
   avg = ydata[0]
   for i in range(len(ydata)):
       # j indexes into ammended_ydata
       j = i + window
       avg = avg - ammended_ydata[j-window]/window + ammended_ydata[j]/window
       smoothed_ydata.append(avg)
   return smoothed_ydata



# smoothed version

ydatanyc_all  = [x/N1 for x in ydatanyc_all_raw]
ydatanassau_all  = [x/N2 for x in ydatanassau_all_raw]
ydatasuffolk_all  = [x/N3 for x in ydatasuffolk_all_raw]


ydatanyc_all = smooth_data(ydatanyc_all,5)
ydatanassau_all = smooth_data(ydatanassau_all,5)
ydatasuffolk_all = smooth_data(ydatasuffolk_all,5)



total_data_size = len(ydatasuffolk_all)
training_size = 45
# training data
xdata = xdata_all[0:training_size]
ydatasuffolk = ydatasuffolk_all[0:training_size]
ydatanassau = ydatanassau_all[0:training_size]
ydatanyc = ydatanyc_all[0:training_size]


xdata = np.array(xdata, dtype=np.float64)
ydatasuffolk = np.array(ydatasuffolk, dtype=np.float64)
ydatanassau = np.array(ydatanassau, dtype=np.float64)
ydatanyc = np.array(ydatanyc, dtype=np.float64)


# test data
xdata_test = xdata_all[training_size : total_data_size]
ydatasuffolk_test = ydatasuffolk_all[training_size : total_data_size]
ydatanassau_test = ydatanassau_all[training_size : total_data_size]
ydatanyc_test = ydatanyc_all[training_size : total_data_size]


real_Is = np.stack((ydatanyc, ydatanassau, ydatasuffolk)).T


var_num = 18


def sepir_model(y, x, beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,
                   gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                   alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3):
   assert(len(y) == var_num)
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


   sum_ISPS1 = S1 * (beta11*(C1+I1) + gamma11*P1 + beta12*(C2+I2) + gamma12*P2 + beta13*(C3+I3) + gamma13*P3);
   dS1dt = -sum_ISPS1;
   dE1dt = sum_ISPS1 - alpha1 * E1;
   dP1dt = alpha1 * E1 - tau1 * P1;
   dC1dt = tau1*P1 - 1*C1;
   dI1dt = 1*C1 - delta1 * I1;
   dR1dt = delta1 * I1;


   sum_ISPS2 = S2 * (beta21*(C1+I1) + gamma21*P1 + beta22*(C2+I2) + gamma22*P2 + beta23*(C3+I3) + gamma23*P3);
   dS2dt = -sum_ISPS2;
   dE2dt = sum_ISPS2 - alpha2 * E2;
   dP2dt = alpha2 * E2 - tau2 * P2;
   dC2dt = tau2*P2 - 1*C2
   dI2dt = 1*C2 - delta2 * I2;
   dR2dt = delta2 * I2;


   sum_ISPS3 = S3 * (beta31*(C1+I1) + gamma31*P1 + beta32*(C2+I2) + gamma32*P2 + beta33*(C3+I3) + gamma33*P3);
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
   solution = integrate.odeint(sepir_model, ((N1-303)/N1, 300/N1, 2/N1, 1/N1, 0, 0, (N2-33)/N2, 30/N2, 2/N2, 1/N2, 0, 0, (N3-33)/N3, 30/N3, 2/N3, 1/N3, 0, 0),
                           x, args=(beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,
                                   gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                                   alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3))
   return solution




def residual(z):
   global xdata
   beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,  \
       gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33, \
       alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3 \
       = z
   solution = solve_ode(xdata, beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33,
                   gamma11, gamma12, gamma13, gamma21, gamma22, gamma23, gamma31, gamma32, gamma33,
                   alpha1, alpha2, alpha3, tau1, tau2, tau3, delta1, delta2, delta3)
   all_Is = solution[:, [3, 9, 15]]
   diff_Is = all_Is - real_Is
   sum1 = 0
   sum2 = 0
   sum3 = 0
   for i in range(0,training_size):
       sum1 += (all_Is[i][0] - real_Is[i][0])*(all_Is[i][0] - real_Is[i][0])
       sum2 += (all_Is[i][1] - real_Is[i][1])*(all_Is[i][1] - real_Is[i][1])
       sum3 += (all_Is[i][2] - real_Is[i][2])*(all_Is[i][2] - real_Is[i][2])
   
   
   return math.sqrt(sum1*sum1 + sum2*sum2 + sum3*sum3)




result = optimize.minimize(residual,
                             (0.2, 0.02, 0.01, 0.03, 0.18, 0.01, 0.02, 0.02, 0.17,   0.36, 0.04, 0.01, 0.2, 0.35, 0.02, 0.02, 0.03, 0.35,   0.4, 0.4, 0.4, 0.33, 0.33, 0.33, 0.142, 0.142, 0.142),
                             bounds=[(0.1, 0.33), (0.0, 0.06), (0.0, 0.06), (0.0, 0.06), (0.0, 0.23), (0.0, 0.06), (0.0, 0.06),
                                     (0.0, 0.06), (0.0, 0.23), (0.2, 0.5), (0.0, 0.08), (0.0, 0.08), (0.0, 0.08), (0.2, 0.5),
                                     (0.0, 0.08), (0.0, 0.08), (0.0, 0.08), (0.2, 0.5), (0.35, 0.35), (0.35, 0.35), (0.35, 0.35),
                                     (0.33, 0.35), (0.33, 0.35), (0.33, 0.35), (0.15, 0.17), (0.15, 0.17), (0.15, 0.17)], method='L-BFGS-B')


fitted = solve_ode(xdata, *result.x)


fitted_nyc = fitted[:,3]
fitted_nas = fitted[:,9]
fitted_suf = fitted[:,15]


last_fitted_y = fitted[-1, :]


def get_pred_values(x, y_initial, *params):
   assert(len(y_initial) == var_num)
   # print(f"y_initial={y_initial}")


   result = []
   # start from the last fitted values for the variables, generate predicted values
   # step by step by calculating the delta of all variables using the model function
   # and fitted parameters
   y_prev = y_initial
   for i in x:
       dy = sepir_model(y_prev, x, *params)
       # print(f"dy={dy}")
       # dy = sepir_model(y_prev, [], *params)    # note: we use a dummy [] for x
       y = y_prev + dy
       result.append(y)
       # print(f"y={y}")
       y_prev = y


   return np.array(result)


pred_values = get_pred_values(xdata_test, last_fitted_y, *result.x)


nyc_predict = pred_values[:, 3]
nas_predict = pred_values[:, 9]
suf_predict = pred_values[:, 15]






fig, axes = plt.subplots(3, figsize=(10,12))


def plot_ax(ax, ydataall_arg, fitted_arg, predicted_arg, ylabel_tag):
   ax.plot(xdata_all, ydataall_arg, 'o')
   ax.plot(xdata, fitted_arg, 'g')
   ax.plot(xdata_test, predicted_arg, 'r')
   ax.set_xlabel("Number of Days")
   ax.set_ylabel(f"{ylabel_tag} Cases")
   ax.vlines(training_size, 0, 1, transform=ax.get_xaxis_transform(), linestyles='dotted', colors='darkorange')
   scaling = len(fitted_arg)/len(ydataall_arg)
   lower_data_mark_y = 0.1
   higher_data_mark_y = 0.8
   ax.annotate('', (scaling-0.2, lower_data_mark_y), xycoords='axes fraction',
               xytext=(scaling, lower_data_mark_y), textcoords='axes fraction',
               arrowprops=dict(color='darkorange', shrink=0.05, width=1, headlength=5,headwidth=5),
               fontsize=8,
               horizontalalignment='right', verticalalignment='top')
   ax.text(scaling-0.1, lower_data_mark_y+0.02, 'training', horizontalalignment='center', verticalalignment='bottom',transform=ax.transAxes)
   ax.annotate('', (scaling+0.2, higher_data_mark_y), xycoords='axes fraction',
               xytext=(scaling, higher_data_mark_y), textcoords='axes fraction',
               arrowprops=dict(color='darkorange', shrink=0.05, width=1, headlength=5,headwidth=5),
               fontsize=8,
               horizontalalignment='right', verticalalignment='top')
   ax.text(scaling+0.1, higher_data_mark_y+0.02, 'testing', horizontalalignment='center', verticalalignment='bottom',transform=ax.transAxes)
   # ax.arrow(41, 5, 10, 0, lw=1, head_width=100, head_length=2, facecolor='black')
   ax.legend(['Published', 'Fitted', 'Predicted'], title='Case Number Measures',alignment='left')



def upscale_by_population(ydata, n):
    return [x*n for x in ydata]
ydatanyc_all  = [x*N1 for x in ydatanyc_all]
fitted_nyc  = [x*N1 for x in fitted_nyc]
nyc_predict  = [x*N1 for x in nyc_predict]
ydatanassau_all = upscale_by_population(ydatanassau_all, N2)
fitted_nas = upscale_by_population(fitted_nas, N2)
nas_predict = upscale_by_population(nas_predict, N2)
ydatasuffolk_all = upscale_by_population(ydatasuffolk_all, N3)
fitted_suf = upscale_by_population(fitted_suf, N3)
suf_predict = upscale_by_population(suf_predict, N3)

plot_ax(axes[0], ydatanyc_all, fitted_nyc, nyc_predict, "NYC")
plot_ax(axes[1], ydatanassau_all, fitted_nas, nas_predict, "Nassau")
plot_ax(axes[2], ydatasuffolk_all, fitted_suf, suf_predict, "Suffolk")


plt.figure(figsize=(40,36),dpi=100)
plt.show()





