################
###   模型定义
# -------------
from model.SUNet_restormerQE import SUNet_model
from net_stdf import MFVQE
opts_dict = {
        'network':{
            'radius': 3,
            'swinunet': {
                'patch_size': 4,
                'in_chans': 7,
                'out_chans': 32,
                'embed_dim': 48,
                'depths': [2, 2, 2],
                'num_heads': [2, 2, 2],
                'window_size': 8,
                'mlp_ratio': 1.,
                'qkv_bias': False,
                'qk_scale': 2,
                },
            'qenet': {
                'dim': 32,
                'num_blocks': [4],
                'heads': [1],
                },
        }}
# opts_dict = {
#         'network':{
#             'radius': 3,
#             'stdf': {
#                 'in_nc': 1,
#                 'out_nc': 64,
#                 'nf': 64,
#                 'nb': 3,
#                 'base_ks': 3,
#                 'deform_ks': 3,
#             },
#             'qenet': {
#                 'in_nc': 64,
#                 'out_nc': 1,
#                 'nf': 64,
#                 'nb': 16,
#                 'base_ks': 3,
#             },
#         }}
# net = MFVQE(opts_dict=opts_dict['network'])
net = SUNet_model(opts_dict=opts_dict['network'])
######################################
type_size = 4  # float
params = list(net.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))
print('Model {} : params: {:4f}M'.format(net._get_name(), k * type_size / 1000 / 1000))

######################################