'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import numpy as np
import scipy.io as sio
from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, perform_tilt


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data

@register_operator(name='simu_cassi')
class CASSIOperator(LinearOperator):
    def __init__(self, device, mask_path=None, mask_type='Phi_PhiPhiT', input_setting='Y'):
        '''
        mask_path: path to the mask file (.mat)
        mask_type: type of the mask Phi_PhiPhiT or PhiTPhi
        input_setting: Y or X
        image_size: size of the image (assumed square)
        '''
        self.device = device
        self.mask_type = mask_type
        self.input_setting = input_setting
        self.mask_path = mask_path
        self.mask3d_batch, self.input_mask = self.init_mask()
        # print(f'the shape of mask3d_batch is {self.mask3d_batch.shape}')
    def init_mask(self,nC=28):
        mask = sio.loadmat(self.mask_path + '/mask.mat')['mask']
        mask = np.tile(mask[:,:, np.newaxis, np.newaxis], (1, 1, nC, 1))  # H*W*nC*1
        mask = np.transpose(mask, (3, 2, 0, 1))  # 1*nC*H*W
        mask3d_batch = torch.from_numpy(mask).cuda().float()
        if self.mask_type == 'Phi':
            shift_mask3d_batch = self.shift(mask3d_batch)
            input_mask = shift_mask3d_batch
        elif self.mask_type == 'Phi_PhiPhiT':
            Phi_batch, Phi_s_batch = self.generate_shift_masks(self.mask_path)
            input_mask = (Phi_batch, Phi_s_batch)
        elif self.mask_type == 'Mask':
            input_mask = mask3d_batch
        elif self.mask_type == None:
            input_mask = None
        return mask3d_batch, input_mask
    def shift(self,data,step=2):
        [bs, nC, row, col] = data.shape
        output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[:, i, :, step * i:step * i + col] = data[:, i, :, :]
        return output
    def shift_back(self,inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
        [bs, row, col] = inputs.shape
        nC = 28
        output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
        return output
    def generate_shift_masks(self,mask_path):
        mask_3d_shift = sio.loadmat(mask_path + '/mask_3d_shift.mat')['mask_3d_shift']
        mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
        mask_3d_shift = torch.from_numpy(mask_3d_shift)
        [nC, H, W] = mask_3d_shift.shape
        Phi_batch = mask_3d_shift.expand([1, nC, H, W]).cuda().float()
        Phi_s_batch = torch.sum(Phi_batch**2,1)  # [b,256,310]
        Phi_s_batch[Phi_s_batch==0] = 1
        return Phi_batch, Phi_s_batch
    def gen_meas_torch(self,data_batch,mask3d_batch,Y2H=True,mul_mask=False):
        nC = data_batch.shape[1] #28
        temp = self.shift(mask3d_batch * data_batch, 2) # [bs,28,256,310]
        meas = torch.sum(temp, 1) # [bs,256,310]
        if Y2H:
            meas = meas / nC * 2 # [bs,256,310]
            H = self.shift_back(meas) # [bs,28,256,256]
            if mul_mask:
                HM = torch.mul(H, mask3d_batch)
                return HM
            return H
        return meas
    def forward(self, data, **kwargs):
        print(f'the shape of data is {data.shape}')
        if self.input_setting == 'H':
            input_meas = self.gen_meas_torch(data, self.mask3d_batch,Y2H=True, mul_mask=False)
        elif self.input_setting == 'HM':
            input_meas = self.gen_meas_torch(data, self.mask3d_batch, Y2H=True, mul_mask=True)
        elif self.input_setting == 'Y':
            input_meas = self.gen_meas_torch(data, self.mask3d_batch, Y2H=False, mul_mask=True) #[1,256,310]
        return input_meas
    def transpose(self, data, **kwargs):
        return self.shift_back(data) #[1,28,256,310]
    def project(self, data, measurement, **kwargs): # data: [1,28,256,256]  measurement: [1,256,310]
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

        

    
@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

@register_operator(name='blind_blur')
class BlindBlurOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        #TODO: faster way to apply conv?:W
        
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img

@register_operator(name='turbulence')
class TurbulenceOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, tilt, **kwargs):
        tilt_data = perform_tilt(data, tilt, image_size=data.shape[-1], device=data.device)
        blur_tilt_data = self.apply_kernel(tilt_data, kernel)
        return blur_tilt_data

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        # self.random_kernel.requires_grad = False
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        for param in blur_model.parameters():
            param.requires_grad = False
        return blur_model
    
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)