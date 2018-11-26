import time
import os

import numpy as np
import math

#warnings.filterwarnings('error')
RESULT_FOLDER = 'Results/'


def psnr(im1,im2,vmax=None):
    """ 
           psnr - compute the Peack Signal to Noise Ratio

           p = psnr(x,y,vmax);

           defined by :
                  p = 10*log10( vmax^2 / |x-y|^2 )
              |x-y|^2 = mean( (x(:)-y(:)).^2 )
              if vmax is ommited, then
                 vmax = max(max(x(:)),max(y(:)))

    """

    if vmax==None :
        m1 = np.max( abs(im1) )
        m2 = np.max( abs(im2) )
        vmax = max(m1,m2)

    d = np.mean( ( im1 - im2 )**2 )

    return 10*math.log10( vmax**2/d )


 
def timeit(method):
    """
        A decorator to measure execution time
    """
 
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
 
        print(method.__name__ + ' has run during ' + str(te-ts) + 's')
        return result
 
    return timed

def handleSeed(method):
    """
        A decorator to handle seed
    """
 
    def wrapper(*args, **kwargs):
        bound_args = inspect.signature(method).bind(*args, **kwargs)
        bound_args.apply_defaults()

        try:
            newSeed = bound_args.arguments['seed']
        
        except Exception as e:
            # The argument 'seed' probably not exist
            newSeed = None
            
        finally:
            curState = np.random.get_state()

        np.random.seed(newSeed)
        result = method(*args, **kwargs)

        if newSeed is not None:
            np.random.set_state(curState)

        return result

    return wrapper


#np.random.seed(0)

def norm2error(mat_Y, mat_D, mat_W):

    mat_err = mat_Y - mat_D.dot(mat_W)

    return np.sqrt(np.sum(np.sum(mat_err**2)))



def isRedundantWrtD(vec_d, mat_D, treshold):
    """
        Input:
            - vec_d must be a Px1 array
            - mat_D must be a PxK array
            - treshold can be any real number, but should be between 0 and 1

        Output:
            - True or False
    """


    vec_norm = np.sqrt(np.sum(mat_D**2, axis=0))
    mat_Dnorm = mat_D / vec_norm[None, :]
    

    vec_d /= np.linalg.norm(vec_d)
    vec_scalProd = np.dot(vec_d.transpose(), mat_Dnorm)
    
    if np.any(vec_scalProd >= treshold):
        return True
    
    return False


def im2Patches(mat_Im, patchSizeL, patchSizeC, step):
    """ im2Patches extracts image into several patches with "sliding" mode, then rearranges patch into column.
    mat_Y=im2Patches(mat_Im, patchSizeL, patchSizeC, step) 
    mat_Im : Image
    PatchSizeL: height (row) of patch
    PatchSizeC: width (column) of patch
    step: the step for sliding
    
    Example:
        mat_A=np.reshape(np.array(range(1,17)),(4,4))
        mat_A
        array([[ 1,  2,  3,  4],
               [ 5,  6,  7,  8],
               [ 9, 10, 11, 12],
               [13, 14, 15, 16]])
        mat_Y=im2Patches(mat_A,2,2,2)
        array([[  1.,   9.,   3.,  11.],
               [  2.,  10.,   4.,  12.],
               [  5.,  13.,   7.,  15.],
               [  6.,  14.,   8.,  16.]])
    """
    nb_rows = patchSizeL*patchSizeC
    nb_col = math.floor( (mat_Im.shape[1] - patchSizeC) / step + 1 ) * math.floor( (mat_Im.shape[0] - patchSizeL) / step + 1 )
    mat_Y = np.zeros((nb_rows,nb_col))
    n=0
    for j in range(0,mat_Im.shape[1]-patchSizeC+1,step):
         for i in range(0,mat_Im.shape[0]-patchSizeL+1,step):
             patch=mat_Im[i:i+patchSizeL,j:j+patchSizeC]
             mat_Y[:,n]=np.reshape(patch,nb_rows) 
             n=n+1
             
    return mat_Y

def patches2Ima(mat_Y,sizeIm,patchSizeL,patchSizeC,step):
    """patches2Ima rearranges matrix columns into blocks, then restores an
    images from these blocks.
    mat_Iout=patches2Ima(Y,sizeIm,patchSizeL,patchSizeC,step) 
    mat_Y: matrix columns
    sizeIm: tuple of size of Image
    patchSizeL: height (row) of patch
    patchSizeC: width (column) of patch
    step: the step for sliding between 2 patches

    Example:
        mat_A=np.reshape(np.array(range(1,17)),(4,4))
        mat_A
        array([[ 1,  2,  3,  4],
               [ 5,  6,  7,  8],
               [ 9, 10, 11, 12],
               [13, 14, 15, 16]])
        mat_Y=im2Patches(mat_A,2,2,2)
        mat_Y
        array([[  1.,   9.,   3.,  11.],
               [  2.,  10.,   4.,  12.],
               [  5.,  13.,   7.,  15.],
               [  6.,  14.,   8.,  16.]])
    
        mat_Abis=patches2Ima(mat_Y,(4,4),2,2,2)
        mat_Abis
        array([[  1.,   2.,   3.,   4.],
               [  5.,   6.,   7.,   8.],
               [  9.,  10.,  11.,  12.],
               [ 13.,  14.,  15.,  16.]])
    """
    
    
    mat_Iout=np.zeros(sizeIm)
    mat_Weight=np.zeros(sizeIm)
    mat_pathOnes=np.ones((patchSizeL,patchSizeC))
    n=0
    for j in range(0,sizeIm[1]- patchSizeC+1,step):
         for i in range(0,sizeIm[0]- patchSizeL+1,step):
             mat_Iout[i:i+patchSizeL,j:j+patchSizeC]=mat_Iout[i:i+patchSizeL,j:j+patchSizeC]+np.reshape(mat_Y[:,n],(patchSizeL,patchSizeC))
             mat_Weight[i:i+patchSizeL,j:j+patchSizeC]=mat_Weight[i:i+patchSizeL,j:j+patchSizeC]+ mat_pathOnes
             n=n+1
    
    mat_Weight[mat_Weight==0]=1
    mat_Iout=mat_Iout/mat_Weight
    return mat_Iout
    
    
    
    
    
    
    
    
    
    
    
    
    