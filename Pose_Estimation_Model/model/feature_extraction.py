import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_utils import (
    interpolate_pos_embed,
    get_chosen_pixel_feats,
    sample_pts_feats
)
import vmamba

class ViMBackbone(nn.Module):
    def __init__(self, cfg,) -> None:
        super(ViMBackbone, self).__init__()
        self.cfg = cfg
        self.pretrained = cfg.pretrained

        self.vim = vmamba.Backbone_VSSM( depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer=("ln2d" if True else "ln"), 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
         )

        self.upsample = nn.Sequential(
            nn.Conv2d(1440, 512, kernel_size=2, padding=2),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=2, padding=2),
            nn.GELU(),
        )
           
        if self.pretrained: 
            vim_checkpoint = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'checkpoints',
                'vssm_small_0229_ckpt_epoch_222.pth',
            )
            checkpoint = torch.load(vim_checkpoint, map_location='cpu', weights_only=True)
            print("load pre-trained checkpoint from: %s" % vim_checkpoint)
            checkpoint_model = checkpoint['model']
            state_dict = self.vim.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vim, checkpoint_model)
            self.vim.load_state_dict(checkpoint_model, strict=False)

    def forward(self, x):
        _, _, _, _ = x.size()

        backbone_outs = self.vim(x)
        
        cls_tokens = backbone_outs[-1][:,0,:].contiguous()

        upscaled_features = [
            F.interpolate(backbone_outs[0], size=(56, 56), mode='bilinear', align_corners=False),
            F.interpolate(backbone_outs[1], size=(56, 56), mode='bilinear', align_corners=False),
            F.interpolate(backbone_outs[2], size=(56, 56), mode='bilinear', align_corners=False),
            F.interpolate(backbone_outs[3], size=(56, 56), mode='bilinear', align_corners=False)
        ]

        x = torch.cat(upscaled_features, dim=1)
        x = self.upsample(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x, cls_tokens
    



class FeatureExtraction(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(FeatureExtraction, self).__init__()
        self.npoint = npoint
        self.rgb_net = ViMBackbone(cfg)

    def forward(self, end_points):
        rgb = end_points['rgb']
        rgb_choose = end_points['rgb_choose']
        dense_fm = self.get_img_feats(rgb, rgb_choose)
        dense_pm = end_points['pts']
        assert rgb_choose.size(1) == self.npoint

        if not self.training and 'dense_po' in end_points.keys() and 'dense_fo' in end_points.keys():
            dense_po = end_points['dense_po'].clone()
            dense_fo = end_points['dense_fo'].clone()

            radius = torch.norm(dense_po, dim=2).max(1)[0]
            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            dense_po = dense_po / (radius.reshape(-1, 1, 1) + 1e-6)

        else:
            tem1_rgb = end_points['tem1_rgb']
            tem1_choose = end_points['tem1_choose']
            tem1_pts = end_points['tem1_pts']

            dense_po = torch.cat([tem1_pts], dim=1)
            radius = torch.norm(dense_po, dim=2).max(1)[0]
            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            tem1_pts = tem1_pts / (radius.reshape(-1, 1, 1) + 1e-6)
            dense_po, dense_fo = self.get_obj_feats(
                [tem1_rgb],
                [tem1_pts],
                [tem1_choose]
            )

        return dense_pm, dense_fm, dense_po, dense_fo, radius

    def get_img_feats(self, img, choose):
        return get_chosen_pixel_feats(self.rgb_net(img)[0], choose)

    def get_obj_feats(self, tem_rgb_list, tem_pts_list, tem_choose_list, npoint=None):
        if npoint is None:
            npoint = self.npoint

        tem_feat_list =[]
        for tem, tem_choose in zip(tem_rgb_list, tem_choose_list):
            tem_feat_list.append(self.get_img_feats(tem, tem_choose))

        tem_pts = torch.cat(tem_pts_list, dim=1)
        tem_feat = torch.cat(tem_feat_list, dim=1)

        return sample_pts_feats(tem_pts, tem_feat, npoint)
