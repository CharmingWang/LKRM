import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
# from model.rpn.rpn_fpn import _RPN_FPN
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool,ROIAlignAvg
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_decode
import time
import pdb



class A_compute(nn.Module):
    def __init__(self, input_features, nf=64, ratio=[4, 2, 1]):
        super(A_compute, self).__init__()
        self.num_features = nf
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        #        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), int(nf * ratio[2]), 1, stride=1)
        self.conv2d_4 = nn.Conv2d(int(nf * ratio[2]), 1, 1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cat_feature):
        W1 = cat_feature.unsqueeze(2)#shape[8,128,1,2048]
        W2 = torch.transpose(W1, 1, 2)#shape[8,1,128,2048]
        W_new = torch.abs(W1 - W2)#shape[8,128,128,2048]
        W_new = torch.transpose(W_new, 1, 3)#shape[8,2048,128,128]
        W_new = self.conv2d_1(W_new)#shape[8,256,128,128]
        W_new = self.relu(W_new)
        W_new = self.conv2d_2(W_new)#shape[8,128,128,128]
        W_new = self.relu(W_new)
        W_new = self.conv2d_3(W_new)#shape[8,64,128,128]
        W_new = self.relu(W_new)
        W_new = self.conv2d_4(W_new)#shape[8,1,128,128]
        W_new = W_new.contiguous()
        W_new = W_new.squeeze(1)
        Adj_M = W_new#shape[8,128,128]
        return Adj_M

class Know_Rout_mod(nn.Module):
    def __init__(self, input_features, output_features):
        super(Know_Rout_mod, self).__init__()
        self.input_features = input_features
        self.lay_1_compute_A = A_compute(input_features)
        self.transferW = nn.Linear(input_features, output_features)

    def forward(self, cat_feature):
        cat_feature_stop = Variable(cat_feature.data)
        Adj_M1 = self.lay_1_compute_A(cat_feature_stop)
        # batch matrix-matrix product
        W_M1 = F.softmax(Adj_M1, 2)

        # batch matrix-matrix product
        cat_feature = torch.bmm(W_M1, cat_feature)#shape[8,128,128]*[8,128,2048]
        cat_feature = self.transferW(cat_feature)#shape[8,128,256]

        return cat_feature, Adj_M1

class GraphAttentionLayer(nn.Module):
    """ graph attention layer """
    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.kaiming_uniform_(self.W.data)
        if self.out_features:
            nn.init.xavier_uniform_(self.W.data)

        # nn.init.xavier_uniform(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.kaiming_uniform_(self.a.data)
        if self.out_features:
            nn.init.xavier_uniform_(self.a.data)


        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        # print(h.repeat(1, N).view(N*N, -1).size())
        # paired node feature concate
        if self.out_features:
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1) \
                .view(N, -1, 2 * self.out_features)
        else:
            a_input = torch.tensor([]).cuda()
            print(self.a)


        # print(a_input.size())
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class n_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(n_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class mGATNet(nn.Module):
   

    def __init__(self, input_dim=10,hide_dim=256,out_dim=256,dropout=0.2, alpha=1, nheads=4):
        super(mGATNet, self).__init__()

        self.gat1 = n_GAT(input_dim, hide_dim,hide_dim,dropout, alpha, nheads)
        self.gat2 = n_GAT(hide_dim, out_dim,out_dim,dropout, alpha, nheads)

    def forward(self, feature ,adjacency):
        h = F.relu(self.gat1( feature,adjacency))
        logits = self.gat2(h,adjacency)
        return logits


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
     
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.output_dim:
            init.kaiming_uniform_(self.weight)
        # init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
   
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class scence_box():
    def __init__(self,im_info):
        self.scence_info=im_info
    def get_scence_box(self):
        #print(len(self.scence_info))
        #print(self.scence_info[0])
        scence = []
        for i in range(len(self.scence_info)):
            #print(self.scence_info[i][0])
            scence.append([0,0,0,self.scence_info[0][0],self.scence_info[0][1]])
        #print('scence_box is '+str(scence))
        #print(scence)
        scence_tensor=torch.Tensor(scence)
        return scence_tensor



class GcnNet(nn.Module):
   

    def __init__(self, input_dim=10,hide_dim=256,out_dim=256):
        super(GcnNet, self).__init__()

        self.gcn1 = GraphConvolution(input_dim, hide_dim)
        self.gcn2 = GraphConvolution(hide_dim, out_dim)

    def forward(self, feature ,adjacency):
        h = F.relu(self.gcn1( feature,adjacency))
        logits = self.gcn2(h,adjacency)
        return logits


class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic, cls_a_prob, cls_r_prob, cls_s_prob,cls_s_ac_prob,cls_s_d_prob,cls_s_i_prob, modules_size, modules_exist, hkrms_exist):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.modules_size = modules_size
        self.modules_exist = modules_exist
        self.hkrms_exist = hkrms_exist
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlignAvg((cfg.POOLING_SIZE+1, cfg.POOLING_SIZE+1), 1.0/16.0, 0)
        # self.RCNN_roi_pool = ROIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = ROIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.cls_fully_connect = nn.Linear(2048, 2048)
        self.bbox_fully_connect = nn.Linear(2048, 2048)
        self.spat_bbox_fully_connect = nn.Linear(2048 + (int(self.modules_exist[2] > 0)) * self.modules_size, 2048 + (int(self.modules_exist[2] > 0)) * self.modules_size)
        self.occu_cls_fully_connect = nn.Linear(2048 + (int(self.modules_exist[1] > 0)) * self.modules_size, 2048 + (int(self.modules_exist[2] > 0)) * self.modules_size)
        self.sin_sc_flag = [1, 0]
        if self.modules_exist[0]:
            self.compute_node_f_a =  nn.Linear(2048, 512)
            self.compute_node_f_a2 = nn.Linear(2048, 512)
            self.compute_node_f_a3 = nn.Linear(2048, 512)
            self.one_Know_Rout_mod_a = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_a2 = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_a3 = Know_Rout_mod(512, modules_size)
            self.gt_adj_a = nn.Parameter(torch.from_numpy(cls_a_prob), requires_grad=False)#shape[1001.1001]
            self.gt_adj_a2 = nn.Parameter(torch.from_numpy(cls_a_prob), requires_grad=False)
            self.gt_adj_a3 = nn.Parameter(torch.from_numpy(cls_a_prob), requires_grad=False)
            self.GAT = mGATNet(512,256,modules_size, 0.2, 1, 4)
            self.appearance_transform = nn.Linear(15, 2048)

        if self.modules_exist[1]:
            self.compute_node_f_r = nn.Linear(2048, 512)
            self.compute_node_f_r2 = nn.Linear(2048, 512)
            self.compute_node_f_r3 = nn.Linear(2048, 512)
            self.one_Know_Rout_mod_r = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_r2 = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_r3 = Know_Rout_mod(512, modules_size)
            self.gt_adj_r = nn.Parameter(torch.from_numpy(cls_r_prob), requires_grad=False)
            self.gt_adj_r5 = nn.Parameter(torch.from_numpy(cls_r_prob), requires_grad=False)
            self.gt_adj_r3 = nn.Parameter(torch.from_numpy(cls_r_prob), requires_grad=False)
            self.GAT = mGATNet(512, 256, modules_size, 0.2, 1, 4)

        if self.modules_exist[2]:

            self.compute_node_f_s = nn.Linear(2048, 512)
            self.compute_node_f_s2 = nn.Linear(2048, 512)
            self.compute_node_f_s3 = nn.Linear(2048, 512)
            self.change_shape = nn.Linear(modules_size*3,modules_size)
            self.one_Know_Rout_mod_s = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_s2 = Know_Rout_mod(512, modules_size)
            self.one_Know_Rout_mod_s3 = Know_Rout_mod(512, modules_size)
            self.gt_adj_s = nn.Parameter(torch.from_numpy(cls_s_prob), requires_grad=False)
            self.gt_adj_s2 = nn.Parameter(torch.from_numpy(cls_s_prob), requires_grad=False)
            self.gt_adj_s3 = nn.Parameter(torch.from_numpy(cls_s_prob), requires_grad=False)
            self.gt_adj_s_ac = nn.Parameter(torch.from_numpy(cls_s_ac_prob), requires_grad=False)
            self.gt_adj_s_d = nn.Parameter(torch.from_numpy(cls_s_d_prob), requires_grad=False)
            self.gt_adj_s_i = nn.Parameter(torch.from_numpy(cls_s_i_prob), requires_grad=False)
            self.GCN_ac = GcnNet(2048, 512, self.modules_size)
            self.GCN_d = GcnNet(2048, 512, self.modules_size)
            self.GCN_i = GcnNet(2048, 512, self.modules_size)
            self.GCN_inter = GcnNet(2048, 512, self.modules_size)
            self.feat_change_shape = nn.Linear(2048, 512)
        if self.sin_sc_flag[0] == 1:  #cat用的
            #self.sc_tocat = nn.Linear(2048 , modules_size[3])
            self.sc_forward_fc1 = nn.Linear(2048, 5)
            self.sc_forward_fc2 = nn.Linear(5, 2048)

            #自注意力机制
            self.sc_tocat_atte_wq = nn.Linear(2048,2048,bias=False)
            self.sc_tocat_atte_wk = nn.Linear(2048, 2048, bias=False)
            self.sc_tocat_atte_wv = nn.Linear(2048, 256, bias=False)
            self.sc_tocat_atte_softmax = nn.Softmax(dim=2)
            #通道注意力机制
            #self.channel_atte_fc1 = nn.Conv2d(2048, 2048//16, 1, bias=False)
            #self.channel_atte_relu = nn.ReLU()
            #self.channel_atte_fc2 = nn.Conv2d(2048//16 , 2048, 1, bias=False)
            #self.channel_atte_sigmoid = nn.Sigmoid()
        #####################################
        #####################################
        #####################################
        ##################################### GRU-场景检测
        if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):  #re和cat都用的
            self.gru = nn.GRUCell(input_size=2048,hidden_size=2048).cuda()
        ##################################
        ##################################



    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        # normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_2nd, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_2nd, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_3rd, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_3rd, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # weights_init(self.RCNN_top_2nd, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # weights_init(self.RCNN_top_3rd, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def threshold_calculate(self, relation):

        newrelation = np.zeros((128, 128))

        for i in range(1, len(relation)):
            for j in range(1, len(relation)):
                newrelation[i - 1][j - 1] = relation[i][j]
        newrelation = newrelation.flatten()

        # relation.sort()
        newrelation = list(newrelation)
        listrelation = []
        for i in newrelation:
            if i > 0 and i < 1:
                listrelation.append(i)

        listrelation = torch.tensor(listrelation)

        return torch.mean(listrelation)

    def normalization(self,adjacency, symmetric=True):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        # adjacency += sp.eye(adjacency.shape[0])    # 增加自连�?        #      # degree = np.array(adjacency.sum(1))
        #      # d_hat = sp.diags(np.power(degree, -0.5).flatten())
        #      # return d_hat.dot(adjacency).dot(d_hat)
        # A = A+I
        Ashape = adjacency.shape[0]
        # adjacency = torch.from_numpy(adjacency)
        adjacency = adjacency + torch.eye(Ashape).cuda()
        # 所有节点的�?        
        d = adjacency.sum(1)
        if symmetric:
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(adjacency).mm(D)
        else:
            # D=D^-1
            D = torch.diag(torch.pow(d, -1))
            return D.mm(adjacency)

    # def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
    #     ''' roi pool on pyramid feature maps'''
    #     # do roi pooling based on predicted rois
    #     # print("rois shape",rois.shape)
    #     # print("feat_maps",feat_maps.shape)
    #     img_area = im_info[0][0] * im_info[0][1]
    #     h = rois.data[:, 4] - rois.data[:, 2] + 1
    #     w = rois.data[:, 3] - rois.data[:, 1] + 1
    #     # print(h)
    #     # print(w)
    #
    #     roi_level = torch.log2(torch.sqrt(h * w) / 224.0)
    #     roi_level = torch.round(roi_level + 4)
    #     roi_level[roi_level < 2] = 2
    #     roi_level[roi_level > 5] = 5
    #     # roi_level.fill_(5)
    #     # print("roi_level",roi_level)
    #     if cfg.POOLING_MODE == 'align':
    #         roi_pool_feats = []
    #         box_to_levels = []
    #         for i, l in enumerate(range(2, 6)):
    #             # print(i, l)
    #             # print(roi_level)
    #             if (roi_level == l).sum() == 0:
    #                 continue
    #
    #             idx_l = (roi_level == l).nonzero().squeeze()
    #             # print(idx_l.dim())
    #             # print((idx_l.cpu().numpy()))
    #             if(idx_l.dim()==0):
    #                 idx_l=idx_l.unsqueeze(0)
    #                 # continue
    #                 # print("^^^^^^^^^^^^^^^^^^^^^^",idx_l.dim())
    #             box_to_levels.append(idx_l)
    #             scale = feat_maps[i].size(2) / im_info[0][0]
    #             # self.RCNN_roi_align.scale=scale
    #             feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l],scale)
    #             roi_pool_feats.append(feat)
    #
    #         # print("box_to_levels")
    #         # print(box_to_levels)
    #         roi_pool_feat = torch.cat(roi_pool_feats, 0)
    #         box_to_level = torch.cat(box_to_levels, 0)
    #         idx_sorted, order = torch.sort(box_to_level)
    #         roi_pool_feat = roi_pool_feat[order]
    #
    #     elif cfg.POOLING_MODE == 'pool':
    #         roi_pool_feats = []
    #         box_to_levels = []
    #         for i, l in enumerate(range(2, 6)):
    #             if (roi_level == l).sum() == 0:
    #                 continue
    #             idx_l = (roi_level == l).nonzero().squeeze()
    #             box_to_levels.append(idx_l)
    #             scale = feat_maps[i].size(2) / im_info[0][0]
    #             self.RCNN_roi_pool.scale=scale
    #             feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l])
    #             roi_pool_feats.append(feat)
    #         roi_pool_feat = torch.cat(roi_pool_feats, 0)
    #         box_to_level = torch.cat(box_to_levels, 0)
    #         idx_sorted, order = torch.sort(box_to_level)
    #         roi_pool_feat = roi_pool_feat[order]
    #
    #     return roi_pool_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        list110 = [1, 1, 0]
        list000 = [0, 0, 0]
        list010 = [0, 1, 0]

        base_feat = self.RCNN_base(im_data)

        # feed image data to base model to obtain base feature map
        # Bottom-up
        # c1 = self.RCNN_layer0(im_data)
        # c2 = self.RCNN_layer1(c1)
        # c3 = self.RCNN_layer2(c2)
        # c4 = self.RCNN_layer3(c3)
        # c5 = self.RCNN_layer4(c4)
        # Top-down
        # p5 = self.RCNN_toplayer(c5)
        # p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        # p4 = self.RCNN_smooth1(p4)
        # p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        # p3 = self.RCNN_smooth2(p3)
        # p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        # p2 = self.RCNN_smooth3(p2)

        # p6 = self.maxpool2d(p5)
        #
        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        # mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # print("rois shape stage1:",rois.shape)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        # roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
        base_feat = self.RCNN_base(im_data)
        #if self.modules_exist == list000:
        #    cfg.POOLING_MODE = 'pool'

        scale = base_feat.size(2) / im_info[0][0]
        if cfg.POOLING_MODE == 'align':
            # if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
            # #if self.module_exist[3]:
            #     sc=scence_box(im_info)
            #     sc_box=sc.get_scence_box()
            #     sc_box=torch.unsqueeze(sc_box,dim=1)
            #     sc_box=sc_box.cuda()
            #     sc_box = sc_box[0]
            #     #print(sc_box)  #sc_box为<class 'torch.cuda.FloatTensor'>[2,1,5]
            #     #print(type(rois.data)) #rois.data为<class 'torch.cuda.FloatTensor'>[2,128,5]
            #     #print(type(sc_box))
            #
            #     rois.data=torch.cat((sc_box,rois.data),0)  #则新的rois.data为[2，129，5]
            roi_pool_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5),scale)
        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        # pooled_feat = self._head_to_tail(roi_pool_feat)
        pooled_feat = self._head_to_tail(roi_pool_feat)

        # #######################################
        # ####################################### cat的时候sc连在了前边，这儿应该取前面的2x2048，这个应该是有顺序的吧？
        # if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
        #     # if self.module_exist[3]:
        #     scence_feat = pooled_feat[0:batch_size, :]  # [2,2048]
        #     pooled_feat = pooled_feat[batch_size:, :]  # [256,2048]
        #     rois = rois[1:, :]  # [2,128,5]
        # ############################################
        # ############################################
        #     # visual attribute
        # visual_feat = pooled_feat.view(batch_size, -1, 2048).cuda()  # visual_feat [2，128，2048]
        #     # three modules
        # pooled_feat = visual_feat.clone()
        #
        # ########################################
        # ########################################
        # if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
        #     # if self.module_exist[3]:
        #     seq_len = 2
        #     # print('attention please!!!!!!!!!!')
        #     # print(pooled_feat.size()[1])
        #     scence_feat = self.sc_forward_fc1(scence_feat)
        #     scence_feat = self.sc_forward_fc2(scence_feat)
        #     scence_feat = torch.unsqueeze(scence_feat, dim=1)  # scence_feat [2,1,2048]
        #     h_all = []
        #     # print(visual_feat[0])
        #     # print(visual_feat[1])
        #
        #     for t in range(len(visual_feat)):  # 遍历每个bs,len(visual_feat)其实就是batch_size
        #         scence_feat_temp = scence_feat[t]  # scence_feat_temp[1,2048],一个bs中的
        #         scence_feat_tt = scence_feat_temp
        #         for k in range(pooled_feat.size()[1] - 1):  # 将scence_feat_temp变为[128,2048] 每个128都是一样的
        #             scence_feat_temp = torch.cat((scence_feat_temp, scence_feat_tt), dim=0)
        #         scence_feat_v = scence_feat_temp  # scence_feat_v[128,2048]
        #         h1 = visual_feat[t]  # h1[128,2048],同样是从bs（=2）一个个取出来的
        #         for i in range(seq_len):
        #             h1 = self.gru(scence_feat_v, h1)  # 得到的新h1也是[128,2048]
        #         # 此处是将一张图的场景作为输入，一张图得到的候选框特征作为状态，通过GRU来更新候选框特征（visual_feat_v）
        #         h_all.append(h1)  # 列表[x，xx]两个值，第一个x为[128,2048]第一张图的128个候选框特征。第二个xx为[128，2048]第二张图的候选框特征
        #     # h_all = torch.Tensor(h_all)
        #     if len(h_all) == 1:
        #         h_all_tensor = torch.unsqueeze(h_all[0], dim=0)
        #     if len(h_all) >= 2:
        #         flag = 1
        #         for i in range(len(h_all) - 1):
        #             if flag == 1:
        #                 h_all_tensor_now = torch.unsqueeze(h_all[0], dim=0)
        #                 h_all_tensor_next = torch.unsqueeze(h_all[1], dim=0)
        #                 h_all_tensor = torch.cat((h_all_tensor_now, h_all_tensor_next), dim=0)
        #             if flag == 0:
        #                 h_all_tensor_next1 = torch.unsqueeze(h_all[i + 1], dim=0)
        #                 h_all_tensor = torch.cat((h_all_tensor, h_all_tensor_next1), dim=0)
        #             flag = 0
        #
        # if (self.sin_sc_flag[0] == 1 and self.sin_sc_flag[1] == 0):  # re型
        #     pooled_feat = h_all_tensor  # 替代型
        # if (self.sin_sc_flag[0] == 0 and self.sin_sc_flag[1] == 1):  # cat型
        #     # sc_gt_feat = self.sc_tocat(h_all_tensor)
        #     # h_all_tensor = h_all_tensor[0]   #h_all_tensor[1,128,2048]转换为[128，2048]
        #
        #     sc_tocat_q = self.sc_tocat_atte_wq(h_all_tensor)
        #     sc_tocat_k = self.sc_tocat_atte_wk(h_all_tensor)
        #     sc_tocat_v = self.sc_tocat_atte_wv(h_all_tensor)
        #
        #     sc_tocat_u = torch.bmm(sc_tocat_q, sc_tocat_k.transpose(1, 2))  # u[1,128,128]
        #     sc_tocat_u = sc_tocat_u / (math.sqrt(2048))
        #
        #     sc_tocat_A = self.sc_tocat_atte_softmax(sc_tocat_u)
        #
        #     sc_gt_feat = torch.bmm(sc_tocat_A, sc_tocat_v)
        #     pooled_feat = torch.cat((pooled_feat, sc_gt_feat), -1)
        # ########################################
        # ########################################



        if self.modules_exist == list000 and self.hkrms_exist == list010:
            box_pooled_feat = self.bbox_fully_connect(pooled_feat)
            box_pooled_feat = F.relu(box_pooled_feat)
            bbox_pred = self.RCNN_bbox_pred(box_pooled_feat)
        else:
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if bbox_pred.size(0) == 1:
            bbox_pred = bbox_pred[0]

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
        if self.modules_exist == list000 and self.hkrms_exist == list010:
            cls_pooled_feat = self.cls_fully_connect(pooled_feat)
            cls_pooled_feat = F.relu(cls_pooled_feat)
            cls_score = self.RCNN_cls_score(cls_pooled_feat)
        else:
            cls_score = self.RCNN_cls_score(pooled_feat)

        if cls_score.size(0) == 1:
            cls_score = cls_score[0]
        cls_prob = F.softmax(cls_score, 1)
        cls_pro_1st = cls_prob
        cls_index1 = torch.max(cls_prob, 1)[1]
        cls_index1 = cls_index1.unsqueeze(0)


        # print(cls_prob)
        # print("*******************cls prob shape",cls_prob.shape)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_cls_2nd = 0
        RCNN_loss_bbox_2nd = 0
        RCNN_loss_cls_3rd = 0
        RCNN_loss_bbox_3rd = 0
        adja3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()
        adjr3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()
        adjs3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()

        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        bbox_pred_1st = bbox_pred
        
        if self.training:
            rois_label = rois_label.view(batch_size, -1)






        # 2nd-----------------------------
        # decode
        if True:
            rois = bbox_decode(rois, bbox_pred, batch_size, self.class_agnostic, self.n_classes, im_info, self.training,
                               cls_prob)

            if self.training:
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, stage=2)
                rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data
                index_ = rois_label.long()

                rois = rois.view(-1, 5)
                rois_label = rois_label.view(-1).long()
                gt_assign = gt_assign.view(-1).long()
                pos_id = rois_label.nonzero().squeeze()
                gt_assign_pos = gt_assign[pos_id]
                rois_label_pos = rois_label[pos_id]
                rois_label_pos_ids = pos_id

                rois_pos = Variable(rois[pos_id])
                rois = Variable(rois)
                rois_label = Variable(rois_label)

                rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))






            else:
                rois_label = None
                gt_assign = None
                rois_target = None
                rois_inside_ws = None
                rois_outside_ws = None
                rpn_loss_cls = 0
                rpn_loss_bbox = 0
                rois = rois.view(-1, 5)
                pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
                # print(pos_id)
                rois_label_pos_ids = pos_id
                rois_pos = Variable(rois[pos_id])
                rois = Variable(rois)
                cls_index1 = cls_index1
                index_ = cls_index1.long()

            if self.hkrms_exist[1]:
                # shape[1,128]
                if self.modules_exist[1] == 2:
                    gt_adj_r5 = Variable(cls_index1.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()# 初始化shape[8,128,128]
                if self.modules_exist[2] == 3:
                    gt_adj_s2 = Variable(cls_index1.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()
                    gt_adj_s_ac = Variable(cls_index1.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()
                    gt_adj_s_d = Variable(cls_index1.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()
                    gt_adj_s_i = Variable(cls_index1.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()

            if self.hkrms_exist[1]:

                if self.modules_exist[1] == 2:
                    for b in range(batch_size):
                        temp = self.gt_adj_r[index_[b], :]
                        temp = temp.transpose(0, 1)[index_[b], :]
                        gt_adj_r5[b] = temp.transpose(0, 1)

                if self.modules_exist[2] == 3:
                    for b in range(batch_size):
                        temp = self.gt_adj_s[index_[b], :]
                        temp = temp.transpose(0, 1)[index_[b], :]
                        gt_adj_s2[b] = temp.transpose(0, 1)
                    for b in range(batch_size):
                        temp = self.gt_adj_s_ac[index_[b], :]
                        temp = temp.transpose(0, 1)[index_[b], :]
                        gt_adj_s_ac[b] = temp.transpose(0, 1)
                    for b in range(batch_size):
                        temp = self.gt_adj_s_d[index_[b], :]
                        temp = temp.transpose(0, 1)[index_[b], :]
                        gt_adj_s_d[b] = temp.transpose(0, 1)
                    for b in range(batch_size):
                        temp = self.gt_adj_s_i[index_[b], :]
                        temp = temp.transpose(0, 1)[index_[b], :]
                        gt_adj_s_i[b] = temp.transpose(0, 1)

            # roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
            # feed pooled features to top model
            if cfg.POOLING_MODE == 'align':
                if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
                    # if self.module_exist[3]:
                    sc = scence_box(im_info)
                    sc_box = sc.get_scence_box()
                    sc_box = torch.unsqueeze(sc_box, dim=1)
                    sc_box = sc_box.cuda()
                    sc_box = sc_box[0]
                    # print(sc_box)  #sc_box为<class 'torch.cuda.FloatTensor'>[2,1,5]
                    # print(type(rois.data)) #rois.data为<class 'torch.cuda.FloatTensor'>[2,128,5]
                    # print(type(sc_box))

                    rois.data = torch.cat((sc_box, rois.data), 0)  # 则新的rois.data为[2，129，5]
                roi_pool_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5), scale)
            elif cfg.POOLING_MODE == 'pool':
                roi_pool_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            pooled_feat = self._head_to_tail_2nd(roi_pool_feat)

            #######################################
            ####################################### cat的时候sc连在了前边，这儿应该取前面的2x2048，这个应该是有顺序的吧？
            if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
                # if self.module_exist[3]:
                scence_feat = pooled_feat[0:batch_size, :]  # [2,2048]
                pooled_feat = pooled_feat[batch_size:, :]  # [256,2048]
                rois = rois[1:, :]  # [2,128,5]
            ############################################
            ############################################
            visual_feat = pooled_feat.view(batch_size, -1, 2048).cuda()  # visual_feat [2，128，2048]
            # three modules
            pooled_feat = visual_feat.clone()

            ########################################
            ########################################
            if (self.sin_sc_flag[0] == 1 or self.sin_sc_flag[1] == 1):
                # if self.module_exist[3]:
                seq_len = 2
                # print('attention please!!!!!!!!!!')
                # print(pooled_feat.size()[1])
                scence_feat = self.sc_forward_fc1(scence_feat)
                scence_feat = self.sc_forward_fc2(scence_feat)
                scence_feat = torch.unsqueeze(scence_feat, dim=1)  # scence_feat [2,1,2048]
                h_all = []
                # print(visual_feat[0])
                # print(visual_feat[1])

                for t in range(len(visual_feat)):  # 遍历每个bs,len(visual_feat)其实就是batch_size
                    scence_feat_temp = scence_feat[t]  # scence_feat_temp[1,2048],一个bs中的
                    scence_feat_tt = scence_feat_temp
                    for k in range(pooled_feat.size()[1] - 1):  # 将scence_feat_temp变为[128,2048] 每个128都是一样的
                        scence_feat_temp = torch.cat((scence_feat_temp, scence_feat_tt), dim=0)
                    scence_feat_v = scence_feat_temp  # scence_feat_v[128,2048]
                    h1 = visual_feat[t]  # h1[128,2048],同样是从bs（=2）一个个取出来的
                    for i in range(seq_len):
                        h1 = self.gru(scence_feat_v, h1)  # 得到的新h1也是[128,2048]
                    # 此处是将一张图的场景作为输入，一张图得到的候选框特征作为状态，通过GRU来更新候选框特征（visual_feat_v）
                    h_all.append(h1)  # 列表[x，xx]两个值，第一个x为[128,2048]第一张图的128个候选框特征。第二个xx为[128，2048]第二张图的候选框特征
                # h_all = torch.Tensor(h_all)
                if len(h_all) == 1:
                    h_all_tensor = torch.unsqueeze(h_all[0], dim=0)
                if len(h_all) >= 2:
                    flag = 1
                    for i in range(len(h_all) - 1):
                        if flag == 1:
                            h_all_tensor_now = torch.unsqueeze(h_all[0], dim=0)
                            h_all_tensor_next = torch.unsqueeze(h_all[1], dim=0)
                            h_all_tensor = torch.cat((h_all_tensor_now, h_all_tensor_next), dim=0)
                        if flag == 0:
                            h_all_tensor_next1 = torch.unsqueeze(h_all[i + 1], dim=0)
                            h_all_tensor = torch.cat((h_all_tensor, h_all_tensor_next1), dim=0)
                        flag = 0

            if (self.sin_sc_flag[0] == 1 and self.sin_sc_flag[1] == 0):  # re型
                pooled_feat = h_all_tensor  # 替代型
            if (self.sin_sc_flag[0] == 0 and self.sin_sc_flag[1] == 1):  # cat型
                # sc_gt_feat = self.sc_tocat(h_all_tensor)
                # h_all_tensor = h_all_tensor[0]   #h_all_tensor[1,128,2048]转换为[128，2048]

                sc_tocat_q = self.sc_tocat_atte_wq(h_all_tensor)
                sc_tocat_k = self.sc_tocat_atte_wk(h_all_tensor)
                sc_tocat_v = self.sc_tocat_atte_wv(h_all_tensor)

                sc_tocat_u = torch.bmm(sc_tocat_q, sc_tocat_k.transpose(1, 2))  # u[1,128,128]
                sc_tocat_u = sc_tocat_u / (math.sqrt(2048))

                sc_tocat_A = self.sc_tocat_atte_softmax(sc_tocat_u)

                sc_gt_feat = torch.bmm(sc_tocat_A, sc_tocat_v)
                pooled_feat = torch.cat((pooled_feat, sc_gt_feat), -1)
            ########################################
            ########################################
            if self.hkrms_exist[1] == 1:
                visual_feat2 = pooled_feat.view(batch_size, -1, 2048)

                pooled_feat2 = visual_feat2.clone()
                if self.modules_exist[0] == 1:
                    visual_feat_512 = self.compute_node_f_a2(visual_feat2)
                    transfer_feat_a2, adja2 = self.one_Know_Rout_mod_a2(visual_feat_512)
                    pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_a2), -1)  # shape[8,128,2048]cat[8,128,256]
                if self.modules_exist[1] == 1:
                    visual_feat_512 = self.compute_node_f_r2(visual_feat2)
                    transfer_feat_r2, adjr2 = self.one_Know_Rout_mod_r2(visual_feat_512)
                    pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_r2), -1)
                if self.modules_exist[2] == 1:
                    visual_feat_512 = self.compute_node_f_s2(visual_feat2)

                    transfer_feat_s2, adjs2 = self.one_Know_Rout_mod_s2(visual_feat_512)
                    pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_s2), -1)

                # occu_pooled_feat = pooled_feat2


                if self.modules_exist[1] == 2:
                    Average3 = self.threshold_calculate(self.gt_adj_r5)
                    # appearance_feat_b3 = self.appearance_transform(cls_prob_2nd)

                    # Adj_a3 = gt_adj_a3[0, :, :]
                    Adj_r2 = torch.where(gt_adj_r5[0, :, :].cuda() > Average3.cuda(), torch.tensor(1).cuda(),
                                         torch.tensor(0).cuda())
                    # visual_feat_b3 = visual_feat3[0,:,:]
                    visual_feat_512 = self.compute_node_f_r2(visual_feat2)
                    visual_feat_512 = visual_feat_512[0, :, :]
                    transfer_feat_r2 = self.GAT(visual_feat_512, Adj_r2)
                    transfer_feat_r2 = transfer_feat_r2.view(batch_size, -1, self.modules_size)
                    pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_r2), -1)
                occu_pooled_feat = pooled_feat2

                if self.modules_exist[2] == 3:
                    ########################使用GCN处理隐含空间知识，取topk个空间关系##################################
                    top = 50
                    Z_feat = self.feat_change_shape(pooled_feat2)
                    Z_feat_T = torch.transpose(Z_feat, 1, 2)
                    adjs_inter = torch.bmm(Z_feat, Z_feat_T)

                    for b in range(batch_size):  # 取每个batch单独进行操作
                        adjs_inter_single = adjs_inter[b, :, :].clone()
                        # adjs_inter_single_new = adjs_inter[b, :, :].clone()
                        adjs_inter_single_new = F.normalize(adjs_inter_single, p=2, dim=1)
                        adjs_inter_single_new2 = adjs_inter_single_new.clone()

                        for i in range(len(adjs_inter_single[0])):
                            list_adjs, adj_index = torch.sort(adjs_inter_single_new[i], descending=True)  # 取前topK个值
                            one = torch.ones_like(adjs_inter_single[i])
                            zero = torch.zeros_like(adjs_inter_single[i])
                            adjs_inter_single_new2[i] = torch.where(adjs_inter_single[i] > list_adjs[50], one, zero)

                            # for j in range(len(adjs_inter_single_new[1])):
                            #     if adjs_inter_single_new[i][j] < list_adjs[top]:
                            #         adjs_inter_single_new[i][j] = 0
                            #     else:
                            #         adjs_inter_single_new[i][j] = 1# 将topk以为的值设为50


                        # Average_s_inter = self.threshold_calculate(adjs_inter_single_new)
                        #
                        # Adj_s_inter = torch.where(adjs_inter_single_new > Average_s_inter, torch.tensor(1).cuda(),
                        #                           torch.tensor(0).cuda())
                        Adj_s_inter = self.normalization(adjs_inter_single_new2)
                        visual_feat_b2 = visual_feat2[b, :, :]
                        transfer_feat_s_inter = self.GCN_inter(visual_feat_b2, Adj_s_inter)
                        transfer_feat_s_inter = transfer_feat_s_inter.view(batch_size, -1, self.modules_size)

                        # pooled_feat2 = torch.cat((visual_feat2, transfer_feat_s_inter), -1)
                        # spat_pooled_feat = pooled_feat2
                        pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_s_inter), -1)
                        spat_pooled_feat = torch.cat((visual_feat2, transfer_feat_s_inter), -1)
                    ########################使用GAT处理隐含空间知识，##################################
                    # top = 50
                    # Z_feat = self.feat_change_shape(pooled_feat2)
                    # Z_feat_T = torch.transpose(Z_feat, 1, 2)
                    # adjs_inter = torch.bmm(Z_feat, Z_feat_T)
                    # for b in range(batch_size):  # 取每个batch单独进行操作
                    #     adjs_inter_single = adjs_inter[b, :, :].clone()
                    #     # adjs_inter_single_new = adjs_inter[b, :, :].clone()
                    #     adjs_inter_single_new = F.normalize(adjs_inter_single,p = 2, dim = 1)
                    #     visual_feat_b2 = visual_feat2[b, :, :]
                    #     transfer_feat_s_inter = self.GAT(visual_feat_b2, adjs_inter_single_new)
                    #     transfer_feat_s_inter = transfer_feat_s_inter.view(batch_size, -1, self.modules_size)
                    #
                    #     # pooled_feat2 = torch.cat((visual_feat2, transfer_feat_s_inter), -1)
                    #     # spat_pooled_feat = pooled_feat2
                    #     pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_s_inter), -1)
                    #     spat_pooled_feat = torch.cat((visual_feat2, transfer_feat_s_inter), -1)


                    # Average_ac = self.threshold_calculate(self.gt_adj_s_ac)
                    # # Adj_s_ac = gt_adj_s_ac[0, :, :]
                    # Adj_s_ac = torch.where(gt_adj_s_ac[0, :, :].cuda() > Average_ac.cuda(), torch.tensor(1).cuda(),
                    #                        torch.tensor(0).cuda())
                    # visual_feat_b2 = visual_feat2[0, :, :]
                    # Adj_s_ac = self.normalization(Adj_s_ac)
                    # transfer_feat_s_ac = self.GCN_ac(visual_feat_b2, Adj_s_ac)
                    # transfer_feat_s_ac = transfer_feat_s_ac.view(batch_size, -1, self.modules_size)
                    #
                    # Average_d = self.threshold_calculate(self.gt_adj_s_d)
                    # # Adj_s_d = gt_adj_s_d[0, :, :]
                    # Adj_s_d = torch.where(gt_adj_s_d[0, :, :].cuda() > Average_d.cuda(), torch.tensor(1).cuda(),
                    #                       torch.tensor(0).cuda())
                    # visual_feat_b2 = visual_feat2[0, :, :]
                    # Adj_s_d = self.normalization(Adj_s_d)
                    # transfer_feat_s_d = self.GCN_d(visual_feat_b2, Adj_s_d)
                    # transfer_feat_s_d = transfer_feat_s_d.view(batch_size, -1, self.modules_size)
                    #
                    # Average_i = self.threshold_calculate(self.gt_adj_s_i)
                    # # Adj_s_i = gt_adj_s_i[0, :, :]
                    # Adj_s_i = torch.where(gt_adj_s_i[0, :, :].cuda() > Average_i.cuda(), torch.tensor(1).cuda(),
                    #                       torch.tensor(0).cuda())
                    # visual_feat_b2 = visual_feat2[0, :, :]
                    # # Adj_s_i = Adj_s_i.float()
                    # Adj_s_i = self.normalization(Adj_s_i)
                    # transfer_feat_s_i = self.GCN_i(visual_feat_b2, Adj_s_i)
                    # transfer_feat_s_i = transfer_feat_s_i.view(batch_size, -1, self.modules_size)
                    #
                    # concat_feature = torch.cat((transfer_feat_s_ac, transfer_feat_s_d), -1)
                    # concat_feature = torch.cat((concat_feature, transfer_feat_s_i), -1)
                    # concat_feature = concat_feature.view(-1, self.modules_size * 3)
                    # spatial_out_feature = self.change_shape(concat_feature)
                    # spatial_out_feature = spatial_out_feature.view(batch_size, -1, self.modules_size)
                    # # spatial_out_feature = transfer_feat_s_ac + transfer_feat_s_d + transfer_feat_s_i
                    #
                    # # pooled_feat3 = torch.cat((pooled_feat3, spatial_out_feature), -1)
                    # pooled_feat2 = torch.cat((pooled_feat2, spatial_out_feature), -1)
                    # spat_pooled_feat = torch.cat((visual_feat2, spatial_out_feature), -1)




                yinzi2 = int(self.modules_exist[0] > 0) + int(self.modules_exist[1] > 0) + int(
                    self.modules_exist[2] > 0)
                pooled_feat = pooled_feat2.contiguous().view(-1, 2048 + self.modules_size * yinzi2)

            # compute bbox offset
            # 控制使用哪种模型
            if self.hkrms_exist[0]:#010
                bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat)
            else:
                if self.modules_exist[2] == 3:#003
                    # bbox_pooled_feat = self.spat_bbox_fully_connect(spat_pooled_feat)
                    bbox_pred = self.RCNN_bbox_pred_2nd(spat_pooled_feat)
                else:
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat)

            if bbox_pred.size(0) == 1:
                bbox_pred = torch.squeeze(bbox_pred)
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                rois_label.long().view(rois_label.size(0), 1, 1).expand(
                                                    rois_label.size(0),
                                                    1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            # compute object classification probability
            if self.hkrms_exist[0]:
                cls_score = self.RCNN_cls_score_2nd(pooled_feat)
            else:
                if self.modules_exist[1] == 2:
                    # cls_pooled_feat = self.occu_cls_fully_connect(occu_pooled_feat)
                    cls_score = self.RCNN_cls_score_2nd(occu_pooled_feat)
                else:
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat)

            if cls_score.size(0) == 1:
                cls_score = torch.squeeze(cls_score)
            cls_prob_2nd = F.softmax(cls_score, 1)
            cls_index2 = torch.max(cls_prob_2nd, 1)[1]
            cls_index2 = cls_index2.unsqueeze(0)

            RCNN_loss_cls_2nd = 0
            RCNN_loss_bbox_2nd = 0
            RCNN_loss_cls_3rd = 0
            RCNN_loss_bbox_3rd = 0
            adja3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()
            adjr3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()
            adjs3_loss = Variable(torch.zeros(1), requires_grad=False).cuda()

            if self.training:
                # loss (cross entropy) for object classification
                RCNN_loss_cls_2nd = F.cross_entropy(cls_score, rois_label)
                # loss (l1-norm) for bounding box regression
                RCNN_loss_bbox_2nd = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

                if self.hkrms_exist[1] == 1:
                    if self.modules_exist[0] == 1:
                        gt_adja2_nograd = gt_adj_a2.detach()
                        adja3_loss = F.smooth_l1_loss(adja2, gt_adja2_nograd)

                    if self.modules_exist[1] == 1:
                        gt_adjr2_nograd = gt_adj_r5.detach()
                        adjr3_loss = F.smooth_l1_loss(adjr2, gt_adjr2_nograd)

                    if self.modules_exist[2] == 1:
                        gt_adjs2_nograd = gt_adj_s2.detach()
                        adjs3_loss = F.smooth_l1_loss(adjs2, gt_adjs2_nograd)

            rois = rois.view(batch_size, -1, rois.size(1))
            # cls_prob_2nd = cls_prob_2nd.view(batch_size, -1, cls_prob_2nd.size(1))  ----------------not be used ---------
            bbox_pred_2nd = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

            if self.training:
                rois_label = rois_label.view(batch_size, -1)

        if not self.training:
            if self.modules_exist != list000:
                cls_prob_3rd_avg = (cls_pro_1st + cls_prob_2nd) / 2
                # cls_prob_3rd_avg = cls_prob_2nd
            else:
                cls_prob_3rd_avg = cls_pro_1st
                bbox_pred_2nd = bbox_pred_1st
        else:
            if self.modules_exist != list000:
                cls_prob_3rd_avg = cls_prob_2nd
            else:
                cls_prob_3rd_avg = cls_pro_1st
                bbox_pred_2nd = bbox_pred_1st

        if self.training:
            return rois, cls_prob_3rd_avg, bbox_pred_2nd, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, rois_label, adja3_loss, adjr3_loss, adjs3_loss

        return rois, cls_prob_3rd_avg, bbox_pred_2nd, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, rois_label