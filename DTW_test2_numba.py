from numba import jit

@jit(nopython=True, parallel=True)
def dtw(X,Y):
    import numpy as np
    X = np.array(X)
    Y = np.array(Y)
    n = len(X)
    m = len(Y)
    C = np.zeros((n+1,m+1)) # cost matrix
    D = np.zeros((n+1,m+1)) # accumulated cost matrix
    
    # Build the cost matrix and accumulated cost matrix
    # C and D are initialized with the extended row and column with infinity values:
    # D(n,0) = D(0,m) = inf for n in [1,n] and m in [1,m]. Note D(0,0) = 0!
    for i in range(1,n+1):
        C[i,0] = np.inf
        D[i,0] = np.inf
    
    for i in range(1,m+1):
        C[0,i] = np.inf
        D[0,i] = np.inf
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            # Build C
            C[i,j] = cost(X[i-1],Y[j-1])
            
            # Initialize D
            if i == 1 and j == 1:
                D[i,j] = C[i,j]
            elif i == 1:
                D[i,j] = C[i,j] + C[i,j-1]
            elif j == 1:
                D[i,j] = C[i,j] + C[i-1,j]
            else: # Build D
                D[i,j] = C[i,j] + np.min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
    
    # Find optimal warping path
    # input: D, output: p*
    
    # Initialize p, p is a list of (x,y) indices of matrix D[x,y] = DTW(x,y)
    p = [] # Reversed warping path. Warping path is computed in reverse order!
    p.append([n,m]) # Boundary condition at ending
    
    i,j = n,m # initialize indices
    while True:
        if (i,j) == (1,1): # end condition
            break
        elif i == 1:
            p.append([1,j-1])
            j = j-1
        elif j == 1:
            p.append([i-1,1])
            i = i-1
        else:
            argmin = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]]) # this may not be unique!
            (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][argmin]
            p.append( [i,j] ) 
            
    p.reverse() # reverse p back
    
    return(p,C,D)
## Start
import os
os.chdir('C:\\Users\\James\\OneDrive\\PythonFiles\\scripts\\DynamicTimeWarping')

## Load data
filePath = 'N:\\HVAC_ModelicaModel_Data\\NOAA_WeatherData'
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
# X = [1,5,33,12,43,21,56,23,50,23,11,56,24,7,23,41,55,33,17,8]
# Y = [5,6,32,55,4,10,59,64,7,15,82,32,15,44,63,8,71,22,54,44]

# X = [0,1,0,0,0,0,1,0,0,0]
# Y = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]

# X = np.random.randint(0,20,size = 40)
# Y = np.random.randint(0,20,size = 40)

# N = 500
# X = np.random.normal(0,1,N)
# Y = np.random.normal(0,1,N)

# X = np.cos(np.linspace(0,2*np.pi,24))
# Y = np.sin(np.linspace(0,2*np.pi,24) + 1)

# Low freq + high freq + noise
N = 1000
periods = 3
shift = 2
X = 40*np.sin(np.linspace(0,2*np.pi,N)) + 15*np.sin(np.linspace(0,2*np.pi,N))# + np.random.normal(0,1,N)
Y = 40*np.sin(np.linspace(0,2*np.pi,N)) + 10*np.sin(np.linspace(0,2*np.pi*periods,N)+shift)# + np.random.normal(0,1,N)

## Compute DTW
# from dtw import dtw, cost
from dtw import cost,warpXY,fitX2Y,fitY2X,warpMatrix
import time
time_start = time.time()


p,C,D = dtw(X,Y)

Xw1,Yw1 = warpXY(X,Y,p)

Xw = fitX2Y(X,Y,p)

Yw = fitY2X(X,Y,p)

M = warpMatrix(X,Y,p)

distance = D[-1,-1]
print(distance)

dtime = time.time() - time_start
print(dtime)

## Plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

## Time series plot with warping
fig = plt.figure()
# fig.add_subplot(411)
ax1 = plt.subplot(411)
plt.plot(range(len(X)),X,'b')
plt.plot(range(len(Y)),Y,'r')

for s in p:
    tx,ty = s[0]-1, s[1]-1 # time step w.r.t x and y
    x,y = X[tx],Y[ty]
    plt.plot([tx,ty],[x,y], 'g',alpha = 0.3,linewidth = 0.8) # plot lines


# Warp both X and Y
ax2 = plt.subplot(412)
plt.plot(range(len(Xw1)),Xw1,'b')
plt.plot(range(len(Yw1)),Yw1,'r')
# Fit X to Y
ax3 = plt.subplot(413,sharex=ax1)
plt.plot(range(len(Xw)),Xw,'b--') # warped
plt.plot(range(len(Y)),Y,'r') # unwarped
# Fit Y to X
ax4 = plt.subplot(414,sharex=ax1)
plt.plot(range(len(X)),X,'b') # unwarped
plt.plot(range(len(Yw)),Yw,'r--') # warped


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
ax1.plot(range(1,len(X)+1),X,'b')

# Y
ax2.plot(Y,range(1,len(Y)+1),'r')
# reverse axis
# ax2.plot(Y,range(len(Y),0,-1),'r')
d = 1
ax2.set_xlim(max(Y)+d,min(Y)-d)

# warping path
ax3.plot(np.matrix(p)[:,0],np.matrix(p)[:,1], 's')
# Cost matrix
# xygrid = np.meshgrid(range(len(X)),range(len(Y)))
plt.contourf(C.T,cmap = plt.cm.gray) # Transpose C!!!
plt.colorbar()
# ax3.imshow(C)

# Padding
plt.subplots_adjust(top=0.92, bottom=0.08,
                    left=0.10, right=0.95,
                    hspace=0.25, wspace=0.35)

plt.tight_layout(pad = 0.8)



## Matrix plots
# Optimal warping path
# plt.matshow(M)

# # C matrix
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

















