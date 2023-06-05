import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import cv2

def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true):
    return np.mean(np.abs(pred-true), axis=(0, 1)).sum()

def test_MAE():
    # (200, 10, 1, 256, 256)
    pred = np.random.rand(200, 10, 1, 256, 256)
    true = np.random.rand(200, 10, 1, 256, 256)
    x = np.mean(np.abs(pred-true), axis=(0, 1)).sum() / (pred.shape[-1] * pred.shape[-2])
    print(x)

if __name__ == '__main__':
    test_MAE()


def MSE(pred, true):
    return np.mean((pred-true)**2, axis=(0, 1)).sum()


# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def SHARP(pred):
    # print(pred.shape) # (1, 10, 1, 256, 256)
    frm = pred[0]
    frm = np.uint8(frm * 255)
    for i in range(10):
        sharp = np.max(cv2.convertScaleAbs(cv2.Laplacian(frm[i], cv2.CV_8U, 2)))
    
    return sharp/10

def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])

    sharp = SHARP(pred)
    
    mae = MAE(pred*255, true*255) / (pred.shape[-1] * pred.shape[-2])
    mse = MSE(pred*255, true*255) / (pred.shape[-1] * pred.shape[-2])

    
    if return_ssim_psnr:
        ssim, psnr = 0, 0
        
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
                psnr += PSNR(pred[b, f], true[b, f])

        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])

        return mae, mse, ssim, psnr, sharp
    else:
        return mae, mse, sharp

