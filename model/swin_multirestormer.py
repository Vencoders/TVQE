import argparse
import time
import os.path as op
import torch
import torch.nn as nn
import yaml
import os

# from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import utils
from model.SUNet_detail_JVT import SUNet
from model.restormer_arch import TransformerBlock, RestormerQE


def receive_arg():
    """Process all hyper-parameters and experiment settings.

    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='../option_QP27.yml',
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

    # opts_dict['test']['restore_iter'] = int(
    #     opts_dict['test']['restore_iter']
    #     )

    return opts_dict

class SUNet_model(nn.Module):
    def __init__(self, opts_dict):
        super(SUNet_model, self).__init__()
        self.swin_unet = SUNet(patch_size=opts_dict['swinunet']['patch_size'],
                               in_chans=opts_dict['swinunet']['in_chans'],
                               out_chans=opts_dict['swinunet']['out_chans'],
                               embed_dim=opts_dict['swinunet']['embed_dim'],
                               depths=opts_dict['swinunet']['depths'],
                               num_heads=opts_dict['swinunet']['num_heads'],
                               window_size=opts_dict['swinunet']['window_size'],
                               mlp_ratio=opts_dict['swinunet']['mlp_ratio'],
                               qkv_bias=opts_dict['swinunet']['qkv_bias'],
                               qk_scale=opts_dict['swinunet']['qk_scale'],
                               drop_rate=0.,
                               attn_drop_rate=0.,
                               drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm,
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False,
                               final_upsample="Dual up-sample")
        self.qenet = RestormerQE(
            dim=opts_dict['qenet']['dim'],
            # num_blocks=opts_dict['qenet']['num_blocks'],
            heads=opts_dict['qenet']['heads'],
            )
    def forward(self, x):
        x1,x2,x3,x4 = self.swin_unet(x)
        x5 = self.qenet(x1,x2,x3,x4)
        x5 += x[:, [3], ...]  # res: add middle frame
        return x5

if __name__ == '__main__':
    opts_dict = receive_arg()
    device= 'cuda'
    height = 1280
    width = 720
    x = torch.randn((1, 7, height, width)).to(device)
    model = SUNet_model(opts_dict=opts_dict['network']).to(device)
    # model = RestormerQE().cuda()
    # print(model)
    # flops, params = profile(model, (x,))
    # print('flops: ',flops/10 ** 12)
    # print('{:>16s} : {:<.4f} [K]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 3))
    # print('input image size: (%d, %d)' % (height, width))
    # # print('FLOPs: %.4f G' % (model.flops() / 1e9))
    # # print('model parameters: ', network_parameters(model))
    # time_start = time.time()
    # with torch.no_grad():
    #     model = model.eval()
    #     x = model(x)
    # print('output image size: ', x.shape)
    #
    # # print(params)
    # time_end = time.time()
    # time_c = time_end - time_start
    # fps = 1/time_c
    # print(x.shape,'time cost:{},s'.format('%.3f' %time_c))
    # print('FPS:{},s'.format('%.3f' % fps))