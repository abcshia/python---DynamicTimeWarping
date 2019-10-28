
# Cost function: Arbitrary choice
def cost(x,y):
    import numpy as np
    return np.abs(x-y)

def dtw(X,Y,cost = cost):
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
            
    # p.reverse() # reverse p back
    # Using np.array is slightly faster for reversing
    p = np.array(p) # convert to np.array
    p = p[::-1] # reverse np.array
    
    
    return(p,C,D)
    

def warpXY(X,Y,p):
    # Warp both X and Y
    Xw = []
    Yw = []
    for xy in p:
        # Xw.append(xy[0]) # indices
        # Yw.append(xy[1])
        Xw.append(X[xy[0]-1])
        Yw.append(Y[xy[1]-1])
    return(Xw,Yw)

def fitX2Y(X,Y,p):
    Xw = []
    y_index = 0
    for xy in p:
        y_new_index = xy[1]
        if y_new_index > y_index:
            Xw.append(X[xy[0]-1])
        y_index = y_new_index
    return(Xw)

def fitY2X(X,Y,p):
    Yw = []
    x_index = 0
    for xy in p:
        x_new_index = xy[0]
        if x_new_index > x_index:
            Yw.append(Y[xy[1]-1])
        x_index = x_new_index
    return(Yw)

def warpMatrix(X,Y,p):
    import numpy as np
    n,m = len(X),len(Y)
    M = np.zeros((n,m))
    for xy in p:
        M[xy[0]-1,xy[1]-1] =1
    return(M)

##
import numpy as np
# from numba import jit
# @jit()
def dtwGC(X,Y,T1,T2,cost = lambda x,y: np.abs(x-y)): # Global Constraints are used to speed up DTW computations
    # import numpy as np
    # from scipy import sparse

    X = np.array(X)
    Y = np.array(Y)
    n = len(X)
    m = len(Y)
    C = np.ones((n+1,m+1))*np.inf # cost matrix
    D = np.ones((n+1,m+1))*np.inf # accumulated cost matrix
    d = min(n+1,m+1)
    # Sanity check:
    if (T1 < 0) or (T2 < 0): 
        print('paramters T1 or T2 < 0')
        return([0,0,0])
    
    if (d + T1 < n + 1) or (d + T2 < m + 1):
        print('paramters T1 or T2 too small\nShould have minimum T1: {}, T2: {}'.format(np.max(n-d+1,0),np.max(m-d+1,0)))
        return([0,0,0])
    
    # Build the cost matrix and accumulated cost matrix
    # C and D are initialized with the extended row and column with infinity values:
    # D(n,0) = D(0,m) = inf for n in [1,n] and m in [1,m]. Note D(0,0) = 0!
    # for i in range(1,n+1):
    #     C[i,0] = np.inf
    #     D[i,0] = np.inf
    # 
    # for i in range(1,m+1):
    #     C[0,i] = np.inf
    #     D[0,i] = np.inf
    
    for ii in range(1,d+1):
        for i in range(max([1,ii-T1]),min([n+1,ii+T1])):
            for j in range(max([1,ii-T2]),min([m+1,ii+T2])):
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
                    D[i,j] = C[i,j] + min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
    
    
    # for i in range(1,n+1):
    #     for j in range(1,m+1):
    #         # Build C
    #         C[i,j] = cost(X[i-1],Y[j-1])
    #         
    #         # Initialize D
    #         if i == 1 and j == 1:
    #             D[i,j] = C[i,j]
    #         elif i == 1:
    #             D[i,j] = C[i,j] + C[i,j-1]
    #         elif j == 1:
    #             D[i,j] = C[i,j] + C[i-1,j]
    #         else: # Build D
    #             D[i,j] = C[i,j] + np.min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
    
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
            # argmin = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]]) # this may not be unique!
            # (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][argmin]
            minVal = min([D[i-1,j-1], D[i-1,j], D[i,j-1]])
            for index,k in enumerate([D[i-1,j-1], D[i-1,j], D[i,j-1]]):
                if minVal == k: break
            (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][index]
            
            p.append( [i,j] ) 
            
    # p.reverse() # reverse p back
    # Using np.array is slightly faster for reversing
    p = np.array(p) # convert to np.array
    p = p[::-1] # reverse np.array
    
    return(p,C,D)
##
import numpy as np
from scipy.sparse import csr_matrix
# from numba import jit
# @jit()
# Sparse matrix version
def dtwGC2(X,Y,T1,T2,cost = lambda x,y: np.abs(x-y)): # Global Constraints are used to speed up DTW computations
    # import numpy as np
    # from scipy import sparse

    X = np.array(X)
    Y = np.array(Y)
    n = len(X)
    m = len(Y)
    C = csr_matrix(np.zeros((n+1,m+1))) # cost matrix
    D = csr_matrix(np.zeros((n+1,m+1))) # accumulated cost matrix
    d = min(n+1,m+1)
    # Sanity check:
    if (T1 < 0) or (T2 < 0): 
        print('paramters T1 or T2 < 0')
        return([0,0,0])
    
    if (d + T1 < n + 1) or (d + T2 < m + 1):
        print('paramters T1 or T2 too small\nShould have minimum T1: {}, T2: {}'.format(np.max(n-d+1,0),np.max(m-d+1,0)))
        return([0,0,0])
    
    # Build the cost matrix and accumulated cost matrix
    # C and D are initialized with the extended row and column with infinity values:
    # D(n,0) = D(0,m) = inf for n in [1,n] and m in [1,m]. Note D(0,0) = 0!
    # for i in range(1,n+1):
    #     C[i,0] = np.inf
    #     D[i,0] = np.inf
    # 
    # for i in range(1,m+1):
    #     C[0,i] = np.inf
    #     D[0,i] = np.inf
    
    for ii in range(1,d):
        # add boundaries
        xmin = max([1,ii-T1])
        xmax = min([n+1,ii+T1])
        ymin = max([1,ii-T2])
        ymax = min([m+1,ii+T2])
        # print(ii,xmin,xmax,ymin,ymax)
        C[xmin-1,ii],D[xmin-1,ii] = np.inf,np.inf
        C[ii,ymin-1],D[ii,ymin-1] = np.inf,np.inf
        if xmax < n: C[xmax+1,ii],D[xmax+1,ii] = np.inf,np.inf
        if ymax < m: C[ii,ymax+1],D[ii,ymax+1] = np.inf,np.inf
        
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
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
                    D[i,j] = C[i,j] + min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
    
 
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
            minVal = min([D[i-1,j-1], D[i-1,j], D[i,j-1]])
            for index,k in enumerate([D[i-1,j-1], D[i-1,j], D[i,j-1]]):
                if minVal == k: break
            (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][index]
            # print(minVal,(i,j))
            
            p.append( [i,j] ) 
            
    # Using np.array is slightly faster for reversing
    p = np.array(p) # convert to np.array
    p = p[::-1] # reverse np.array
    
    
    return(p,C.toarray(),D.toarray())



## Numba version
from numba import jit
import numpy as np
@jit() # let numba decide the compilation options
def dtw_numba(X,Y,cost = lambda x,y: np.abs(x-y)):
    # import numpy as np: Cannot import numpy here!
    X = np.array(X)
    Y = np.array(Y)
    n = len(X)
    m = len(Y)
    C = np.zeros((n+1,m+1)) # cost matrix
    D = np.zeros((n+1,m+1)) # accumulated cost matrix
    # cost = costFunc
    
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
            
    # p.reverse() # reverse p back
    # Using np.array is slightly faster for reversing
    p = np.array(p) # convert to np.array
    p = p[::-1] # reverse np.array
    
    return(p,C,D)

## Numba version 2
# from numba import jit
# import numpy as np
# # @jit(nopython=True,parallel=True)
# @jit(parallel=True)
# def dtw_numba2(X,Y,cost = lambda x,y: np.abs(x-y)):
#     # import numpy as np: Cannot import numpy here!
#     X = np.array(X)
#     Y = np.array(Y)
#     n = len(X)
#     m = len(Y)
#     C = np.zeros((n+1,m+1)) # cost matrix
#     D = np.zeros((n+1,m+1)) # accumulated cost matrix
#     # cost = costFunc
#     
#     # Build the cost matrix and accumulated cost matrix
#     # C and D are initialized with the extended row and column with infinity values:
#     # D(n,0) = D(0,m) = inf for n in [1,n] and m in [1,m]. Note D(0,0) = 0!
#     for i in range(1,n+1):
#         C[i,0] = np.inf
#         D[i,0] = np.inf
#     
#     for i in range(1,m+1):
#         C[0,i] = np.inf
#         D[0,i] = np.inf
#     
#     for i in range(1,n+1):
#         for j in range(1,m+1):
#             # Build C
#             C[i,j] = cost(X[i-1],Y[j-1])
#             
#             # Initialize D
#             if i == 1 and j == 1:
#                 D[i,j] = C[i,j]
#             elif i == 1:
#                 D[i,j] = C[i,j] + C[i,j-1]
#             elif j == 1:
#                 D[i,j] = C[i,j] + C[i-1,j]
#             else: # Build D
#                 D[i,j] = C[i,j] + np.min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
#     
#     # Find optimal warping path
#     # input: D, output: p*
#     
#     # Initialize p, p is a list of (x,y) indices of matrix D[x,y] = DTW(x,y)
#     p = [] # Reversed warping path. Warping path is computed in reverse order!
#     p.append([n,m]) # Boundary condition at ending
#     
#     i,j = n,m # initialize indices
#     while True:
#         if (i,j) == (1,1): # end condition
#             break
#         elif i == 1:
#             p.append([1,j-1])
#             j = j-1
#         elif j == 1:
#             p.append([i-1,1])
#             i = i-1
#         else:
#             argmin = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]]) # this may not be unique!
#             (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][argmin]
#             p.append( [i,j] ) 
#             
#     # p.reverse() # reverse p back
#     # Using np.array is slightly faster for reversing
#     p = np.array(p) # convert to np.array
#     p = p[::-1] # reverse np.array
#     
#     return(p,C,D)





# class DTW:
#     def __initial__(self,X,Y):
#         self.X = X
#         self.Y = Y
#         
#     # Cost function: Arbitrary choice
#     def cost(x,y):
#         import numpy as np
#         return np.abs(x-y)
#     
#     
#     def dtw(X,Y):
#         import numpy as np
#         X = np.array(X)
#         Y = np.array(Y)
#         n = len(X)
#         m = len(Y)
#         C = np.zeros((n+1,m+1)) # cost matrix
#         D = np.zeros((n+1,m+1)) # accumulated cost matrix
#         
#         # Build the cost matrix and accumulated cost matrix
#         # C and D are initialized with the extended row and column with infinity values:
#         # D(n,0) = D(0,m) = inf for n in [1,n] and m in [1,m]. Note D(0,0) = 0!
#         for i in range(1,n+1):
#             C[i,0] = np.inf
#             D[i,0] = np.inf
#         
#         for i in range(1,m+1):
#             C[0,i] = np.inf
#             D[0,i] = np.inf
#         
#         for i in range(1,n+1):
#             for j in range(1,m+1):
#                 # Build C
#                 # C[i,j] = cost(X[i-1],Y[j-1])
#                 C[i,j] = np.abs(X[i-1]-Y[j-1])
#                 
#                 # Initialize D
#                 if i == 1 and j == 1:
#                     D[i,j] = C[i,j]
#                 elif i == 1:
#                     D[i,j] = C[i,j] + C[i,j-1]
#                 elif j == 1:
#                     D[i,j] = C[i,j] + C[i-1,j]
#                 else: # Build D
#                     D[i,j] = C[i,j] + np.min( [ D[i-1,j-1], D[i-1,j], D[i,j-1] ] )
#         
#         # Find optimal warping path
#         # input: D, output: p*
#         
#         # Initialize p, p is a list of (x,y) indices of matrix D[x,y] = DTW(x,y)
#         p = [] # Reversed warping path. Warping path is computed in reverse order!
#         p.append([n,m]) # Boundary condition at ending
#         
#         i,j = n,m # initialize indices
#         while True:
#             if (i,j) == (1,1): # end condition
#                 break
#             elif i == 1:
#                 p.append([1,j-1])
#                 j = j-1
#             elif j == 1:
#                 p.append([i-1,1])
#                 i = i-1
#             else:
#                 argmin = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]]) # this may not be unique!
#                 (i,j) = [(i-1,j-1),(i-1,j),(i,j-1)][argmin]
#                 p.append( [i,j] ) 
#                 
#         p.reverse() # reverse p back
#         
#         return(p,C,D)




