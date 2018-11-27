import cSvaIbpDl

import unittest
import sys
import os

import numpy as np


#sys.path.insert(1, os.path.join(sys.path[0], '..'))


class Test_AlgoReels(unittest.TestCase):

    def test_AlgoReel_mp_AverageErr(self):
        dim_K = 3
        dim_P = 30
        dim_N = 4
        beta = 0.75

        # 1. Creating dictionary
        mat_D = np.zeros((dim_P, dim_K))
        for k in range(dim_K):
            for t in range(dim_P):
                if t < 2*k:
                    mat_D[t, k] = 0
                else:
                    mat_D[t, k] = beta**(t-2*k) * np.sqrt(1 - beta**2)

            mat_D[:, k] /= np.linalg.norm(mat_D[:, k])

        # 2. Create Data
        mat_W = np.zeros((dim_K, dim_N))
        mat_W[0, 0] = 2
        mat_W[1, 1] = 1.4
        mat_W[2, 2] = 1.4
        mat_W[1:3, 3] = np.array([2, 2])

        mat_Y = np.dot(mat_D, mat_W)

        # 3. Start OMP
        doptions = {'verbose':False, 'updt_W':'MP', 'newAtome':'averageErr'}
        
        cSva = cSvaIbpDl.CSvaDl()
        cSva.run(mat_Y, 100, 0.5, 0.2)

        D_hat = cSva.get_dic()
        W_hat = cSva.get_coefs()

        print(D_hat.shape)
        print(W_hat.shape)

        # 4 Check final solution
        err = np.sum((mat_Y - D_hat.dot(W_hat))**2)

        print(err)
        #self.assertTrue(err < 10**(-16))


    def test_AlgoReel_mp_AverageErr(self):
        dim_K = 3
        dim_P = 30
        dim_N = 4
        beta = 0.75

        # 1. Creating dictionary
        mat_D = np.zeros((dim_P, dim_K))
        for k in range(dim_K):
            for t in range(dim_P):
                if t < 2*k:
                    mat_D[t, k] = 0
                else:
                    mat_D[t, k] = beta**(t-2*k) * np.sqrt(1 - beta**2)

            mat_D[:, k] /= np.linalg.norm(mat_D[:, k])

        # 2. Create Data
        mat_W = np.zeros((dim_K, dim_N))
        mat_W[0, 0] = 2
        mat_W[1, 1] = 1.4
        mat_W[2, 2] = 1.4
        mat_W[1:3, 3] = np.array([2, 2])

        mat_Y = np.dot(mat_D, mat_W)

        # 3. Start OMP
        doptions = {'verbose':False, 'updt_W':'MP', 'newAtome':'averageErr'}
        
        cSva = cSvaIbpDl.CSvaDl()
        cSva.run(mat_Y, 100, 0.5, 0.2)

        D_hat = cSva.get_dic()
        W_hat = cSva.get_coefs()

        cSvaInpainting = cSvaIbpDl.CSvaDlInpainting()
        matMask = np.full((dim_P, dim_N), np.int32(1))
        cSvaInpainting.run(mat_Y, matMask, 100, 0.5, 0.2)

        D_hat_inpainting = cSvaInpainting.get_dic()
        W_hat_inpainting = cSvaInpainting.get_coefs()

        print(D_hat.shape)
        print(W_hat.shape)

        print(D_hat_inpainting.shape)
        print(W_hat_inpainting.shape)

        # 4 Check final solution
        err = np.sum((mat_Y - D_hat.dot(W_hat))**2)
        errInpainting = np.sum((mat_Y - D_hat_inpainting.dot(W_hat_inpainting))**2)

        print(err)
        print(errInpainting)
        #self.assertTrue(err < 10**(-16))


if __name__ == '__main__':

    unittest.main()
