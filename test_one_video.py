import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import numpy as np
from collections import OrderedDict
from net_stdf import MFVQE
import utils
from tqdm import tqdm
from utils.YUV_RGB import yuv2rgb
from PIL import Image

ckp_path = 'exp/STDF-R3L/ckp_1.pt'  # trained at QP37, LDP, HM16.5

raw_yuv_path = '/data/dataset/LDV_MFQE/test_18/raw/BasketballPass_416x240_500.yuv'
lq_yuv_path = '/data/dataset/LDV_MFQE/test_18/HM16.5_LDP/QP37/BasketballPass_416x240_500.yuv'
save_old = False
save_current = False
vname = lq_yuv_path.split("/")[-1].split('.')[0]
_,wxh,nfs = vname.split('_')
nfs = int(nfs)
# nfs = min(nfs,150)
w,h = int(wxh.split('x')[0]),int(wxh.split('x')[1])
outlog='./details/'+'STDF_'+vname+'.txt'
def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 64,
            'nf': 64,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 3,
        },
        'qenet': {
            'in_nc': 64,
            'out_nc': 1,
            'nf': 64,
            'nb': 16,
            'base_ks': 3,
        },
    }
    model = MFVQE(opts_dict=opts_dict)
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y ,raw_u ,raw_v = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
    lq_y ,lq_u ,lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)
    # save
    if save_old:
        for idx in range(nfs):
            eR,eG,eB = yuv2rgb(raw_y[idx],raw_u[idx]/255.,raw_v[idx]/255.,h,w)
            # print(raw_u[idx].shape,'vs',eR.shape)
            img = np.stack((eR,eG,eB),-1)
            # print(img.shape)
            # os._exit(233)
            # outputdir = './out/raw/'+vname+"/"
            outputdir = './STDF_out/'+ vname+ "/raw/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            Image.fromarray(img.astype('uint8')).convert('RGB').save(outputdir+str(idx+1).zfill(3)+'.png')

            eR,eG,eB = yuv2rgb(lq_y[idx],lq_u[idx]/255.,lq_v[idx]/255.,h,w)
            # print(raw_u[idx].shape,'vs',eR.shape)
            img = np.stack((eR,eG,eB),-1)
            # print(img.shape)
            # os._exit(233)
            outputdir = './STDF_out/'+ vname+ "/lq/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            Image.fromarray(img.astype('uint8')).convert('RGB').save(outputdir+str(idx+1).zfill(3)+'.png')

    f = open(outlog,"w")
    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    with torch.no_grad():
        for idx in range(nfs):
            # load lq
            idx_list = list(range(idx-3,idx+4))
            idx_list = np.clip(idx_list, 0, nfs-1)
            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()

            # enhance
            enhanced_frm = model(input_data)

            # eval
            gt_frm = torch.from_numpy(raw_y[idx]).cuda()
            batch_ori = criterion(input_data[0, 3, ...], gt_frm)
            batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            msg = str(idx) + " " + str(batch_ori) + " -> " + str(batch_perf) + "\n"
            f.write(msg)
            # save it!
            if save_current:
                eR, eG, eB = yuv2rgb(enhanced_frm.squeeze().cpu().detach().numpy(), lq_u[idx] / 255., lq_v[idx] / 255., h,
                                     w)
                # print(raw_u[idx].shape,'vs',eR.shape)
                img = np.stack((eR, eG, eB), -1)
                # print(img.shape)
                # os._exit(233)
                outputdir = './STDF_out/'+ vname+ "/hq/"
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                # print("??",outputdir)
                Image.fromarray(img.astype('uint8')).convert('RGB').save(outputdir + str(idx + 1).zfill(3) + '.png')
            # display
            pbar.set_description(
                "[{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(batch_ori, unit, batch_perf, unit)
                )
            pbar.update()

        pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
            ori_, unit, enh_, unit, (enh_ - ori_) , unit
            ))
        print('> done.')




if __name__ == '__main__':
    main()
