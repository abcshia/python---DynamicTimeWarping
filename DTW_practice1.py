import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

## DWT

# Sample data:
# X = [1,5,33,12,43,21,56,23,50,23,11,56,24,7,23,41,55,33,17,8]
# Y = [5,6,32,55,4,10,59,64,7,15,82,32,15,44,63,8,71,22,54,44]

X = [0,1,0,0,0,0,1,0,0,0]
Y = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]

# X = np.random.randint(0,20,size = 24)
# Y = np.random.randint(0,20,size = 24)

# X = np.cos(np.linspace(0,2*np.pi,24))
# Y = np.sin(np.linspace(0,2*np.pi,24) + 1)


# X is compared to Y, Y is baseline


# Cost function: Arbitrary choice
def cost(x,y):
    return np.abs(x-y)


# Initialize
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
print(p)


## Plot
fig = plt.figure()
fig.add_subplot(411)
plt.plot(range(len(X)),X,'b')
plt.plot(range(len(Y)),Y,'r')


# Warp both X and Y
Xw = []
Yw = []
for xy in p:
  # Xw.append(xy[0]) # indices
  # Yw.append(xy[1])
  Xw.append(X[xy[0]-1])
  Yw.append(Y[xy[1]-1])


fig.add_subplot(412)
plt.plot(range(len(Xw)),Xw,'b')
plt.plot(range(len(Yw)),Yw,'r')


# Fit X to Y
Xw = []
y_index = 0
for xy in p:
    y_new_index = xy[1]
    if y_new_index > y_index:
        Xw.append(X[xy[0]-1])
    y_index = y_new_index

fig.add_subplot(413)
plt.plot(range(len(Xw)),Xw,'b--') # warped
plt.plot(range(len(Y)),Y,'r') # unwarped


# Fit Y to X
Yw = []
x_index = 0
for xy in p:
    x_new_index = xy[0]
    if x_new_index > x_index:
        Yw.append(Y[xy[1]-1])
    x_index = x_new_index

fig.add_subplot(414)
plt.plot(range(len(X)),X,'b') # unwarped
plt.plot(range(len(Yw)),Yw,'r--') # warped



# matrix plots
# Optimal warping path
M = np.zeros((n,m))
for xy in p:
    M[xy[0]-1,xy[1]-1] =1
plt.matshow(M)

# Heatmaps
# C matrix
hC = plt.matshow(C,cmap=plt.cm.gray)
plt.colorbar(hC)

# D Matrix
hD = plt.matshow(D,cmap=plt.cm.gray)
plt.colorbar(hD)

# Overlay p* on C matrix
Cp = C
for xy in p:
    Cp[xy[0],xy[1]] = 0
hCp = plt.matshow(Cp,cmap=plt.cm.gray)
plt.colorbar(hCp)




plt.show()








