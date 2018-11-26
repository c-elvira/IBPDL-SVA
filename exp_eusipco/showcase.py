import os, sys

import numpy as np
import matplotlib.image as mpimg

import utils

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cSvaIbpDl

SEED_EUSIPCO = 20180304

FOLDER_IMG = '../datas/'

PARAM_NBITER = 150
PARAM_STDNOISE = 40. / 255.
PARAM_LEARNINGOVERLAPPING = 3


# Reproducible research
np.random.seed(SEED_EUSIPCO)


def manipImage(strImgName, lbd1, lbd2, seed=None):
    """

    """

    # 1 Img read
    mat_img = mpimg.imread(FOLDER_IMG + strImgName +'.png')

    # 2 Apply noise 
    std_noise = PARAM_STDNOISE
    mat_noise = np.random.normal(0, std_noise, mat_img.shape)
    mat_Y = utils.im2Patches(mat_img + mat_noise, 8, 8, PARAM_LEARNINGOVERLAPPING)

    # 3. Experiment
    print('Debut manip')
    cSva = cSvaIbpDl.CSvaDl()
    cSva.run(mat_Y, PARAM_NBITER, lbd1, lbd2)
    print('Fin manip')

    matD_hat = cSva.get_dic()
    matW_hat = cSva.get_coefs()
 
    # print
    psnr = utils.psnr(mat_img, 
        utils.patches2Ima(np.dot(matD_hat, matW_hat), mat_img.shape, 8, 8, PARAM_LEARNINGOVERLAPPING)
        )
    print('psnr ' + str(round(psnr, 2)) + ' dB')


if __name__ == '__main__':

    strImgName = 'hill'
    print('Showcase with ' + strImgName)
    manipImage(strImgName, 0.4, 0.2)

