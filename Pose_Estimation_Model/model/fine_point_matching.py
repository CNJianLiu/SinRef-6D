import torch
import torch.nn as nn
from point_mamba import PointMamba

from transformer import SparseToDenseTransformer
from model_utils import aug_fine_pose_noise, aug_pose_noise, compute_feature_similarity, compute_fine_Rt
from loss_utils import compute_correspondence_loss

class FinePointMatching(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(FinePointMatching, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * .02)
        self.pointmamba = PointMamba()

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(SparseToDenseTransformer(
                cfg.hidden_dim,
                num_heads=4,
                sparse_blocks=['self', 'cross'],
                dropout=None,
                activation_fn='ReLU',
                focusing_factor=cfg.focusing_factor,
                with_bg_token=True,
                replace_bg_token=True
            ))
        self.transformers = nn.ModuleList(self.transformers)
        
        self.transformers_f = []
        for _ in range(self.nblock):
            self.transformers_f.append(SparseToDenseTransformer(
                cfg.hidden_dim,
                num_heads=4,
                sparse_blocks=['self', 'cross'],
                dropout=None,
                activation_fn='ReLU',
                focusing_factor=cfg.focusing_factor,
                with_bg_token=True,
                replace_bg_token=True
            ))
        self.transformers_f = nn.ModuleList(self.transformers_f)

    def forward(self, p1, fm, geo1, fps_idx1, p2, fo, geo2, fps_idx2, radius, end_points):
        B = p1.size(0)

        c = torch.mean(p1, dim=1, keepdim=True)
        p1_ = p1 - c
        c = c.squeeze(1)
        f1 = self.pointmamba(p1_)+ self.in_proj(fm)
        f1 = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg
        f2 = self.pointmamba(p2) + self.in_proj(fo)
        f2 = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg
        atten_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, fps_idx1, f2, geo2, fps_idx2)

            if self.training or idx==self.nblock-1:
                atten_list.append(compute_feature_similarity(
                    self.out_proj(f1),
                    self.out_proj(f2),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))
                

        if self.training:
            gt_R = end_points['rotation_label']
            gt_t = end_points['translation_label'] / (radius.reshape(-1, 1)+1e-6)
            init_R, init_t = aug_pose_noise(gt_R, gt_t)
            end_points, _ = compute_correspondence_loss(
                end_points, atten_list, p1, p2, gt_R, gt_t,
                dis_thres=self.cfg.loss_dis_thres,
                loss_str='coarse',
            )
        else:
            init_R, init_t, _ = compute_fine_Rt(
                atten_list[-1], p1, p2,
                None,
            )

        p1f_ = (p1 - init_t.unsqueeze(1)) @ init_R
        f1f = self.pointmamba(p1f_)+self.in_proj(fm) 
        f1f = torch.cat([self.bg_token.repeat(B,1,1), f1f], dim=1) # adding bg
        f2f = self.pointmamba(p2)+self.in_proj(fo) 
        f2f = torch.cat([self.bg_token.repeat(B,1,1), f2f], dim=1) # adding bg
        atten_list_f = []
        for idx in range(self.nblock):
            f1f, f2f = self.transformers_f[idx](f1f, geo1, fps_idx1, f2f, geo2, fps_idx2)

            if self.training or idx==self.nblock-1:
                atten_list_f.append(compute_feature_similarity(
                    self.out_proj(f1f),
                    self.out_proj(f2f),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))

                
        if self.training:
            gt_R = end_points['rotation_label']
            gt_t = end_points['translation_label'] / (radius.reshape(-1, 1)+1e-6)
            aug_fine_pose_noise(gt_R, gt_t)
            end_points, _ = compute_correspondence_loss(
                end_points, atten_list_f, p1, p2, gt_R, gt_t,
                dis_thres=self.cfg.loss_dis_thres,
                loss_str='fine'
            )
        else:
            pred_R, pred_t, _ = compute_fine_Rt(
                atten_list_f[-1], p1, p2,
                None,
            )

        if self.training:
            return end_points

        p3f_ = (p1 - pred_t.unsqueeze(1)) @ pred_R
        f3f = self.pointmamba(p3f_) + self.in_proj(fm)
        f3f = torch.cat([self.bg_token.repeat(B,1,1), f3f], dim=1) # adding bg
        f4f = self.pointmamba(p2) + self.in_proj(fo)
        f4f = torch.cat([self.bg_token.repeat(B,1,1), f4f], dim=1) # adding bg
        atten_list_ff = []
        for idx in range(self.nblock):
            f3f, f4f = self.transformers_f[idx](f3f, geo1, fps_idx1, f4f, geo2, fps_idx2)

            if idx == self.nblock - 1:
                atten_list_ff.append(compute_feature_similarity(
                    self.out_proj(f3f),
                    self.out_proj(f4f),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))
        pred_R, pred_t, _ = compute_fine_Rt(
            atten_list_ff[-1], p1, p2,
            None,
        )

        end_points['pred_R'] = pred_R
        end_points['pred_t'] = pred_t * (radius.reshape(-1, 1)+1e-6)
        end_points['pred_pose_score'] = 1

        if self.return_feat:
            return end_points, self.out_proj(fm), self.out_proj(fo)
        else:
            return end_points

