import os
os.chdir('C:\\Users\\James\\OneDrive\\PythonFiles\\scripts\\DynamicTimeWarping')

## Load data
filePath = r'C:\Users\James\OneDrive\Documents\Berkeley\HVAC\Modelica\TestData' #'N:\\HVAC_ModelicaModel_Data\\NOAA_WeatherData'
fileName = 'SF2010_TemperatureData.csv'
weather_file = filePath + '\\' + fileName
# weatherDate = '2010-06-23'

# load data
import numpy as np
weatherData = np.loadtxt(open(weather_file, "rb"), delimiter=",", skiprows=1,dtype = np.object)
weatherData[-1,-1] = 0 # last cell is empty(missing data)
weatherData = np.array(weatherData[:,1:],dtype = float)
day1 = 120 # weatherData[:,day]
day2 = 280 # weatherData[:,day]

X = weatherData[:,day1]
Y = weatherData[:,day2]


# Sample data:
# X = [1,5,33,12,43,21,4]
# Y = [5,6,32,55,4,10,6,5]
# X = [1,5,33,12,43,21,56,23,50,23,11,56,24,7,23,41,55,33,17,8]
# Y = [5,6,32,55,4,10,59,64,7,15,82,32,15,44,63,8,71,22,54,44]

# X = [0,1,0,0,0,0,1,0,0,0]
# Y = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]

# X = np.random.randint(0,10,size = 20)
# Y = np.random.randint(0,10,size = 20)

# N = 100
# X = np.random.normal(0,1,N)
# Y = np.random.normal(0,1,N)

# X = np.cos(np.linspace(0,2*np.pi,24))
# Y = np.sin(np.linspace(0,2*np.pi,24) + 1)

# Low freq + high freq + noise
# N = 240
# periods = 3
# shift = 2
# X = 40*np.sin(np.linspace(0,2*np.pi,N)) + 15*np.sin(np.linspace(0,2*np.pi,N))# + np.random.normal(0,1,N)
# Y = 40*np.sin(np.linspace(0,2*np.pi,N)) + 10*np.sin(np.linspace(0,2*np.pi*periods,N)+shift)# + np.random.normal(0,1,N)

## Compute DTW
# from dtw import dtw, cost
from dtw import *
# 3rd party fastdtw package
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import time
time_start = time.time()


# for i in range(500):
    # p,C,D = dtw(X,Y) # Normal Version
    # p,C,D = dtw_numba(X,Y,cost) # Numba Version: use numba deafult settings
    # p,C,D = dtw_numba2(X,Y,cost) # Numba Version 2
    # T1 = np.int(len(X)/10)
    # T2 = np.int(len(Y)/10)
    # T1,T2 = 1,4
    # p,C,D = dtwGC(X,Y,T1,T2) # Global Constraint Version
    # distance, p = fastdtw(X,Y,dist=euclidean) # fastdtw package

p,C,D = dtw(X,Y) # Normal Version
# p,C,D = dtw_numba(X,Y,cost) # Numba Version: use numba deafult settings
# p,C,D = dtw_numba2(X,Y,cost) # Numba Version 2

# For some reason, applying global constraint is often slower...... weird
# T1 = np.int(len(X)/10)
# T2 = np.int(len(Y)/10)
# T1,T2 = 2,2
# p,C,D = dtwGC(X,Y,T1,T2) # Global Constraint Version
# p,C,D = dtwGC2(X,Y,T1,T2) # Global Constraint Version, Sparse matrix version

distance = D[-1,-1]

time_start_f = time.time()
distance_f, p_f = fastdtw(X,Y,dist=euclidean) # fastdtw package
# shift p_f's starting index by 1, (0,0) > (1,1)
p_f = list(zip([int(k+1) for k in np.matrix(p_f)[:,0]],[int(k+1) for k in np.matrix(p_f)[:,1]]))

dtime2 = time.time() - time_start_f
dtime = time_start_f - time_start
print('distance = {}, distance_f = {}'.format(distance,distance_f))
print('DTW elapsed time: {} seconds\nFastDTW elapsed time: {}'.format(dtime,dtime2))

Xw1,Yw1 = warpXY(X,Y,p)

Xw = fitX2Y(X,Y,p)

Yw = fitY2X(X,Y,p)

M = warpMatrix(X,Y,p)

Xw1_f,Yw1_f = warpXY(X,Y,p_f)

Xw_f = fitX2Y(X,Y,p_f)

Yw_f = fitY2X(X,Y,p_f)

M_f = warpMatrix(X,Y,p_f)



## Plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

## Time series plot with warping
fig = plt.figure()
# fig.add_subplot(411)
ax1 = plt.subplot(411)
plt.plot(range(len(X)),X,'b',label = 'data1')
plt.plot(range(len(Y)),Y,'r',label = 'data2')
plt.plot([],[],'g',alpha = 0.3,label = 'fits') # dummy plot for legends

for s in p:
    tx,ty = s[0]-1, s[1]-1 # time step w.r.t x and y
    x,y = X[tx],Y[ty]
    plt.plot([tx,ty],[x,y], 'g',alpha = 0.3,linewidth = 0.8) # plot lines
plt.legend(loc = 'best')
plt.xlabel('time step')
plt.ylabel('data[a.u.]')

# Warp both X and Y
ax2 = plt.subplot(412)
plt.plot(range(len(Xw1)),Xw1,'b',alpha = 0.7,label = 'warped data1')
plt.plot(range(len(Yw1)),Yw1,'r',alpha = 0.7,label = 'warped data2')
plt.plot(range(len(Xw1_f)),Xw1_f,'purple',alpha = 0.7,label = 'fast warped data1')
plt.plot(range(len(Yw1_f)),Yw1_f,'orange',alpha = 0.7,label = 'fast warped data2')
plt.legend(loc = 'best')
plt.xlabel('time step')
plt.ylabel('data[a.u.]')

# Fit X to Y
ax3 = plt.subplot(413,sharex=ax1)
plt.plot(range(len(Xw)),Xw,'b--',label = 'warped data1') # warped
plt.plot(range(len(Y)),Y,'r',alpha = 0.5,label = 'unwarped data2') # unwarped
plt.plot(range(len(Xw_f)),Xw_f,color = 'purple',linestyle = '--',label = 'fast warped data1') # warped
plt.legend(loc = 'best')
plt.xlabel('time step')
plt.ylabel('data[a.u.]')

# Fit Y to X
ax4 = plt.subplot(414,sharex=ax1)
plt.plot(range(len(X)),X,'b',label = 'uwarped data1') # unwarped
plt.plot(range(len(Yw)),Yw,'r--',alpha = 0.7,label = 'warped data2') # warped
plt.plot(range(len(Yw_f)),Yw_f,color = 'orange',linestyle = '--',alpha = 0.7,label = 'fast warped data2') # warped
plt.legend(loc = 'best')
plt.xlabel('time step')
plt.ylabel('data[a.u.]')


## Optimal warping path
fig = plt.figure()
ax1 = plt.subplot2grid((7,9),(5,2),rowspan = 2,colspan = 5)
ax2 = plt.subplot2grid((7,9),(0,0),rowspan = 5,colspan = 2)
ax3 = plt.subplot2grid((7,9),(0,2),rowspan = 5,colspan = 6, sharex = ax1,sharey = ax2)
# ax4 = plt.subplot2grid((7,9),(0,8),rowspan = 5,colspan = 6, sharey = ax2)
# ax5 = plt.subplot2grid((7,9),(0,2),rowspan = 5,colspan = 6, sharex = ax1,sharey = ax2)
# ax1 = plt.subplot(224)
# ax2 = plt.subplot(221)
# ax3 = plt.subplot(222)

# # for plot rotation
# from matplotlib.transforms import Affine2D
# import mpl_toolkits.axisartist.floating_axes as floating_axes
# 
# plot_extents = 0, len(Y), min(Y), max(Y)
# transform = Affine2D().rotate_deg(90)
# helper = floating_axes.GridHelperCurveLinear(transform, plot_extents)
# ax2 = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
# # floating_axes.FloatingSubplot(ax2, 11flo1, grid_helper=helper)

# Yreverse = list(Y)
# Yreverse = np.array(Yreverse)

# X
ax1.plot(range(1,len(X)+1),X,'b',label = 'data1')
ax1.legend(loc = 'best')


# Y
ax2.plot(Y,range(1,len(Y)+1),'r',label = 'data2')
# reverse axis
# ax2.plot(Y,range(len(Y),0,-1),'r')
d = 1
ax2.set_xlim(max(Y)+d,min(Y)-d)
ax2.legend(loc = 'best')

# warping path
ax3.plot(np.matrix(p)[:,0],np.matrix(p)[:,1], 's',
         alpha = 0.8,label = 'Optimal warp path')
ax3.plot(np.matrix(p_f)[:,0],np.matrix(p_f)[:,1], 'o',
         color = 'orange',alpha = 0.5,label = 'FastDTW path')         
# ax3.plot(np.matrix(p_f)[:,0]+1,np.matrix(p_f)[:,1]+1, 'o',
#          color = 'orange',alpha = 0.5,label = 'FastDTW path')
ax3.legend(loc = 'best')
ax3.axes.set_xlim(0.5,len(X)+0.5)
ax3.axes.set_ylim(0.5,len(Y)+0.5)

# Cost matrix
# xygrid = np.meshgrid(range(len(X)),range(len(Y)))

# imshow method
# cax = ax3.imshow(C.T, cmap = plt.cm.Wistia)
# ax3.invert_yaxis() # plt.gca().invert_yaxis()
# fig.colorbar(cax)

# contour plot
cax = ax3.contourf(C.T,cmap = plt.cm.gray) # Transpose C!!!
fig.colorbar(cax)
# plt.contourf(C.T,cmap = plt.cm.gray) # Transpose C!!!
# plt.colorbar()


# Padding
plt.subplots_adjust(top=0.92, bottom=0.08,
                    left=0.10, right=0.95,
                    hspace=0.25, wspace=0.35)

plt.tight_layout(pad = 0.8)



## Matrix plots
# Optimal warping path
# plt.matshow(M)

# C matrix
# hC = plt.matshow(C,cmap=plt.cm.gray)
# plt.colorbar(hC)
# 
# # D Matrix
# hD = plt.matshow(D,cmap=plt.cm.gray)
# plt.colorbar(hD)
# 
# # Overlay p* on C matrix
# Cp = C
# for xy in p:
#     Cp[xy[0],xy[1]] = 0
# hCp = plt.matshow(Cp,cmap=plt.cm.gray)
# plt.colorbar(hCp)

plt.show()

