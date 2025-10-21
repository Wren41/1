import argparse
import os
import yaml

import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from ddim_sampler import *
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import clear_color, mask_generator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_img(path):
    data = sio.loadmat(path)
    if "img_expand" in data:
        img = data['img_expand'] / 65536.
    elif "img" in data:
        img = data['img'] / 65536.
    elif 'HSI' in data:
        img = data['HSI']
        print('loading KAIST_CVPR2021 dataset')
        img[img < 0] = 0
        img[img > 1] = 1
        img = img.astype(np.float32)    
        img = img/img.max() * 0.8
    return img

def to_img(data,save_path,is_3d=False,is_01=True):
    data = data.detach().cpu().numpy().squeeze()
    if not is_01:
        data = (data + 1) / 2
    if is_3d:
        figure, ax = plt.subplots(1,3, figsize=(10,4))
        for i in range(3):
            ax[i].imshow(data[i,:,:], cmap='gray')
            ax[i].axis('off')
            ax[i].set_title('band {}'.format(i+1))          
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        figure, ax = plt.subplots(4,7, figsize=(12,7))
        for i in range(4):
            for j in range(7):
                ax[i,j].imshow(data[i*7+j,:,:], cmap='gray')
                ax[i,j].axis('off')
                ax[i,j].set_title('band {}'.format(i*7+j+1))    
        plt.tight_layout()  
        plt.savefig(save_path, dpi=300) 
        plt.close()


class initial_x(nn.Module):
    def __init__(self):
        super(initial_x, self).__init__()
        self.function =  nn.Conv2d(28, 28, 1, padding=0, bias=True)
    def forward(self,y):
        out = self.function(y)
        return out
    
def save_E(E,path):
    E = E.cpu().detach().numpy().squeeze()
    plt.figure()
    plt.imshow(E, cmap='jet')
    plt.axis('off')
    plt.colorbar()      
    plt.title('Spectral Eigenvectors')
    plt.savefig(path, dpi=300)   
    plt.close()
def save_template(img_tensor, path):
    img = img_tensor.detach().cpu().numpy().squeeze()
    figure,axs = plt.subplots(4,7, figsize=(10,4))
    for a in range(4):
        for b in range(7):
            axs[a,b].imshow(img[a*7+b,:,:],vmin=0,vmax=1)
            axs[a,b].axis('off')
            axs[a,b].set_title('band {}'.format(a*7+b+1))
    plt.tight_layout()
    plt.savefig(path,dpi=300)
    plt.close()
def save_y_template(img, path):
    img = img.detach().cpu().numpy().squeeze()
    row, col = img.shape
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')      
    plt.title(f'{row}x{col} Measurement')
    plt.savefig(path, dpi=300)   
    plt.close()

class SpectralLoss(nn.Module):
    def __init__(self):
        super(SpectralLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def sam_loss(self,X,Y):
        """
        计算光谱角度映射SAM损失
        X, Y: 输入张量 [B, C, H, W] (C=28为波段数)
        """
        # 展平空间维度
        X_flat = X.view(X.size(0), X.size(1), -1)  # [B, C, H*W]
        Y_flat = Y.view(Y.size(0), Y.size(1), -1)  # [B, C, H*W]
        
        # 计算点积和范数
        dot_product = torch.sum(X_flat * Y_flat, dim=1)  # [B, H*W]
        norm_X = torch.norm(X_flat, dim=1)  # [B, H*W]
        norm_Y = torch.norm(Y_flat, dim=1)  # [B, H*W]
        
        # 计算余弦相似度
        cos_sim = dot_product / (norm_X * norm_Y + 1e-8)  # 防止除零
        # 计算SAM损失
        sam_loss = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)).mean()
        return sam_loss
    def forward(self,x,y):
        loss = self.mse_loss(x, y)
        sam_loss = self.sam_loss(x, y)
        # print('Loss: {:.6f}, SAM Loss: {:.6f}'.format(loss.item(), sam_loss.item()))
        return 0.9 * loss + 0.1 * sam_loss    
    

def dmplug(model, scheduler, logdir, img='00000', eta=0, lr=1e-2, dataset='celeba',img_model_config=None,task_config=None,device='cuda'):
    dtype = torch.float32
    gt_img_path = './data/{}/{}.mat'.format(dataset,img)
    gt_img = np.array(load_img(gt_img_path))
    # 插值到256*256
    ref = torch.Tensor(gt_img.transpose(2, 0, 1)).to(device) # [28,H,W]
    ref = F.interpolate(ref.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)# [28,256,256]
    ref_numpy = np.transpose(ref.cpu().numpy(),(1,2,0))# [256,256,28]
    # 转换成适合DMPlug的形式
    x = np.transpose(ref_numpy, (2, 0, 1)) # [28,256,256]
    ref_img = torch.Tensor(x).to(dtype).to(device).unsqueeze(0) # [1,28,256,256]
    to_img(ref_img, os.path.join(logdir, '28_gt.png'), is_3d=False)
    ref_img.requires_grad = False
    ch,ms = ref.shape[0],ref.shape[2]
    STEP = 2
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    # 将真值图像进行退化处理和添加噪声
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    # 生成掩码（如果是inpainting任务），其他任务不需要
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )
        mask = mask_gen(ref_img)
        mask = mask[:, 0, :, :].unsqueeze(dim=0)
    # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)
    else:
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img).unsqueeze(0) # [1,1,256,310]
        plt.imsave(os.path.join(logdir, f'measurement_gt.png'), clear_color(y), cmap='gray')
        y_n = noiser(y)
    # 保存退化图像
    y_n.requires_grad = False
    # plt.imsave(os.path.join(logdir, f'degraded.png'), clear_color(y_n), cmap='gray')
    # DMPlug
    # 生成初始点，损失函数，优化器
    X = torch.randn((1, ch, ms, ms), device=device, dtype=dtype, requires_grad=True)
    criterion = torch.nn.MSELoss().to(device)
    # 在DMPlug的扩散模型训练过程中，这个优化器会通过多次迭代逐步调整噪声张量Z，使其从随机噪声逐渐转变为目标图像，同时最小化损失函数。
    params_group1 = {'params': X, 'lr': lr}
    optimizer = torch.optim.Adam([params_group1])
    # 训练迭代
    epochs = 5000 # SR, inpainting: 5,000, nonlinear deblurring: 10,000
    psnrs = []
    ssims = []
    losses = []
    pbar = tqdm(range(epochs), desc="Training")
    for iterator in pbar:
        # DDIM sampling
        model.eval()
        optimizer.zero_grad()
        
        # 使用梯度检查点技术
        HSI = torch.zeros((1, ch, ms, ms), device=device, dtype=dtype)
        
        # 为每个波段创建自定义函数，使用checkpoint来节省显存
        def ddim_sampling_for_band(band_idx):
            if band_idx*3+3 < ch:
                Z = X[:, band_idx*3:band_idx*3+3, :, :] #从[1,1,256,310]中提取对应波段的测量值
            else:
                Z = X[:, ch-3:ch, :, :]

            # print(Y_hat.shape)
            x_t = None
            for i, tt in enumerate(scheduler.timesteps):
                t = (torch.ones(1) * tt).cuda()
                if i == 0:
                    noise_pred = model(Z, t)
                else:
                    noise_pred = model(x_t, t)  #(1,6,256,256)
                noise_pred = noise_pred[:, :3]
                if i == 0:
                    x_t = scheduler.step(noise_pred, tt, Z, return_dict=True, 
                                        use_clipped_model_output=True, eta=eta).prev_sample

                else:
                    x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, 
                                        use_clipped_model_output=True, eta=eta).prev_sample
            
            x_hat = torch.clamp(x_t, -1, 1)
            x_hat = (x_hat + 1) / 2  # to [0,1]
            return x_hat
        # 对每个波段应用梯度检查点
        loss2 = 0
        for band in tqdm(range(10), desc="Sampling Bands"):
            b = 0
            # 使用torch.utils.checkpoint来节省显存
            x_hat = torch.utils.checkpoint.checkpoint(
                ddim_sampling_for_band, 
                torch.tensor(band, device=device),
                preserve_rng_state=True,
                use_reentrant=False
            )
            if b==0:
                to_img(x_hat, f'/mnt/vdb/isalab102/DMPlug-main/template1/rec_band{band}.png', is_3d=True, is_01=True)
            if band != 9:
                HSI[:, band*3:band*3+3, :, :] = x_hat
            else:
                HSI[:, ch-1, :, :] = x_hat[:,2,:,:]
        
        output = HSI
        if measure_config['operator']['name'] == 'inpainting':
            loss = criterion(operator.forward(output, mask=mask), y_n)
        else:
            y_hat = operator.forward(output).unsqueeze(0)

            save_y_template(y_hat, f'/mnt/vdb/isalab102/DMPlug-main/template1/rec_meas.png')
            # loss = criterion(y_hat, y_n) 
            loss = criterion(y_hat, y_n)
        if iterator < 100 and iterator % 10 == 0:
            save_template(output, os.path.join(logdir, f'output_{iterator}.png'))
        elif iterator > 100 and iterator % 100 == 0:
            save_template(output, os.path.join(logdir, f'output_{iterator}.png'))
        # 更新进度条显示loss
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'epoch': f'{iterator+1}/{epochs}'
        })
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ############### Evaluation ################
        with torch.no_grad():
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy = np.transpose(output_numpy, (1, 2, 0))
            # calculate psnr
            # print(ref_numpy.shape, output_numpy.shape)
            tmp_psnr = peak_signal_noise_ratio(ref_numpy, output_numpy)
            psnrs.append(tmp_psnr)
            # calculate ssim
            tmp_ssim = structural_similarity(ref_numpy, output_numpy, channel_axis=2, data_range=1)
            ssims.append(tmp_ssim)
            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy
            if (iterator+1) % 100 == 0:
                print('Iter {}: PSNR: {}, SSIM: {}'.format(iterator, tmp_psnr, tmp_ssim))
    ############### Save Results ################
    figure,axs = plt.subplots(4,7, figsize=(10,4))
    for a in range(4):
        for b in range(7):
            axs[a,b].imshow(best_img[:,:,a*7+b],vmin=0,vmax=1)
            axs[a,b].axis('off')
            axs[a,b].set_title('band {}'.format(a*7+b+1))
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'rec_img.png'),dpi=300)
    plt.close()

    plt.plot(np.array(losses), label='all')
    plt.legend()
    plt.title('Min Loss: {}'.format(np.min(np.array(losses))))
    plt.savefig(os.path.join(logdir, f'loss.png'))
    plt.close()

    plt.plot(np.array(psnrs))
    plt.title('Max PSNR: {}'.format(np.max(np.array(psnrs))))
    plt.savefig(os.path.join(logdir, f'psnr.png'))
    plt.close()

    psnr_res = np.max(psnrs)
    ssim_res = np.max(ssims)
    loss_res = np.min(losses)
    # lpips_res = np.min(lpipss)
    print('PSNR: {}, SSIM: {}'.format(psnr_res, ssim_res))
    with open(os.path.join(logdir, 'results.txt'), 'a') as f:
        f.write('PSNR: {}, SSIM: {},Loss:{}\n'.format(psnr_res, ssim_res, loss_res))




def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="logdir",
        default="./results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="KAIST_CVPR2021"  # cave_1024_28, celeba,KAIST_CVPR2021
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast sampling",
        default=3
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="lr of z",
        default=0.01
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="?",
        help="super_resolution,inpainting,nonlinear_deblur,simu_cassi",
        default='simu_cassi'
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        help="model name",
        default='imagenet256'
    )
    # parser.add_argument(
    #     "--img",
    #     type=int,
    #     nargs="?",
    #     help="image id",
    #     default=0
    # )
    return parser
def torch_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    os.makedirs('/mnt/vdb/isalab102/DMPlug-main/template1/', exist_ok=True)
    torch_seed(123)
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    print(f"当前GPU索引: {torch.cuda.current_device()}")
    opt, unknown = parser.parse_known_args()
    img_model_config = 'configs/model_config_{}.yaml'.format(opt.model)
    task_config = 'configs/tasks/{}_config.yaml'.format(opt.task) # task config = super-resolution, inpainting, nonlinear deblurring
    img_model_config = load_yaml(img_model_config) 
    model = create_model(**img_model_config)
    model = model.to(device)
    model.eval()
    task_config = load_yaml(task_config)
    # Define the DDIM scheduler
    scheduler = DDIMScheduler()
    # 设置时间序列，原文设定的是3步
    scheduler.set_timesteps(opt.custom_steps)
    # 打印出时间步长的三个数值
    print(scheduler.timesteps)
    # img = str(opt.img).zfill(5)
    img_list = os.listdir('./data/{}/'.format(opt.dataset))
    img_list.sort()
    i = 0 
    for img in img_list:
        img = str(img.split('.')[0])
        print('Processing image: {}'.format(img))
        # Create logdir
        logdir = os.path.join(opt.logdir, opt.task, opt.dataset, img)
        os.makedirs(logdir,exist_ok=True)
        # DMPlug
        if i == 3:
            exit()
        else:
            dmplug(model, scheduler, logdir, img=img, eta=opt.eta, lr=opt.lr, dataset=opt.dataset, img_model_config=img_model_config, task_config = task_config, device=device)
        i += 1