import argparse
import time
import os.path as op
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# from model.octconv import OctaveConv
import utils
# from ops.dcn.deform_conv import ModulatedDeformConv
from thop import profile

# ==========
# Spatio-temporal deformable fusion module
# ==========
def receive_arg():
    """Process all hyper-parameters and experiment settings.

    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_1G.yml',
        help='Path to option YAML file.'
    )
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help='Distributed launcher requires.'
    )
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )

    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False

    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
    )

    return opts_dict
class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat


# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
            )

        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
                ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


# ==========
# Quality enhancement module
# ==========

class OctPlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(OctPlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            OctaveConv(in_nc, nf, base_ks,alpha_in=0,alpha_out=0.25),
            # nn.ReLU(inplace=True)
            )

        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                OctaveConv(nf, nf, base_ks,alpha_in=0.25,alpha_out=0.25),
                # nn.ReLU(inplace=True)
                ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = OctaveConv(nf, out_nc, base_ks,alpha_in=0.25,alpha_out=0)

    def forward(self, inputs):
        out,_ = self.in_conv(inputs)
        out,_ = self.hid_conv(out)
        out,_ = self.out_conv(out)
        return out
# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> QE -> residual.
    
    in: (B T C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']
        self.ffnet = STDF(
            in_nc= self.in_nc * self.input_len, 
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            deform_ks=opts_dict['stdf']['deform_ks']
            )
        self.qenet = PlainCNN(
            in_nc=opts_dict['qenet']['in_nc'],  
            nf=opts_dict['qenet']['nf'], 
            nb=opts_dict['qenet']['nb'], 
            out_nc=opts_dict['qenet']['out_nc']
            )

    def forward(self, x):
        out = self.ffnet(x)
        out = self.qenet(out)
        # e.g., B C=[B1 B2 B3 R1 R2 R3 G1 G2 G3] H W, B C=[Y1 Y2 Y3] H W or B C=[B1 ... B7 R1 ... R7 G1 ... G7] H W
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        # print(x[:, 3, ...].shape)
        out += x[:, frm_lst, ...]  # res: add middle frame
        return out

if __name__ == '__main__':
    opts_dict = {
            'network':{
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
            }}
    device= 'cuda:1'
    # opts_dict = receive_arg()
    height = 1280
    width = 720
    x = torch.randn((1, 7, height, width)).to(device)
    model = MFVQE(opts_dict=opts_dict['network']).to(device)
    # model = OctPlainCNN(in_nc=64 ,out_nc=1,nf=48,nb=16,base_ks=3).to(device)
    flops, params = profile(model, inputs=(x,))
    print(flops/10 ** 12,params)
    # print(model)
    print('{:>16s} : {:<.4f} [K]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 3))
    print('input image size: (%d, %d)' % (height, width))
    # print('FLOPs: %.4f G' % (model.flops() / 1e9))
    # print('model parameters: ', network_parameters(model))
    time_start = time.time()
    model.eval()
    with torch.no_grad():
        x = model(x)
    print('output image size: ', x.shape)
    # flops, params = profile(model, (x,))
    # print(flops)
    # print(params)
    time_end = time.time()
    time_c = time_end - time_start
    fps = 1/time_c
    print(x.shape,'time cost:{},s'.format('%.3f' %time_c))
    print('FPS:{},s'.format('%.3f' % fps))