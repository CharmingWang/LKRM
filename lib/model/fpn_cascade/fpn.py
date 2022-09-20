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

class Self_attention(nn.Module):
    def __init__(self, input_features_dim, output_features_dim):
        super(Self_attention, self).__init__()
        self.wk = nn.Linear(input_features_dim, output_features_dim, bias=False)
        self.wq = nn.Linear(input_features_dim, output_features_dim, bias=False)
        self.wv = nn.Linear(input_features_dim, output_features_dim, bias=False)

    def forward(self, cat_fuse_feat):
        wk_feat = self.wk(cat_fuse_feat).unsqueeze(1)
        wq_feat = self.wq(cat_fuse_feat).unsqueeze(1)
        wv_feat = self.wv(cat_fuse_feat).unsqueeze(1)
        wk_feat = torch.transpose(wk_feat, 2, 1)
        att_fuse_feat = torch.bmm(wq_feat, wk_feat)
        att_fuse_feat = att_fuse_feat / sqrt(att_fuse_feat.size(0) * att_fuse_feat.size(1))
        att_fuse_feat = F.softmax(att_fuse_feat, dim=2)
        att_fuse_feat = torch.bmm(att_fuse_feat, wv_feat)
        return att_fuse_feat

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
        self.list023 = [0, 2, 3]
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
        self.bbox_conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bbox_conv2 = nn.Conv2d(1024, 2048, 3, 1, 1)
        self.avgpooling = nn.AvgPool2d(5,3)
        self.cls_fully_connect = nn.Linear(2048, 2048)
        self.bbox_fully_connect = nn.Linear(2048, 2048)
        self.spat_bbox_fully_connect = nn.Linear(2048 + (int(self.modules_exist[2] > 0)) * self.modules_size, 2048 + (int(self.modules_exist[2] > 0)) * self.modules_size)
        self.occu_cls_fully_connect = nn.Linear(2048 + (int(self.modules_exist[1] > 0)) * self.modules_size, 2048 + (int(self.modules_exist[2] > 0)) * self.modules_size)
        self.sc_net_type_flag = [0, 1]
        self.fc7 = nn.Linear(1024 * 7 * 7, 2048)
        self.fc8 = nn.Linear(2048, 2048)
        if self.sc_net_type_flag[1] == 1:
            self.sc_tocat = nn.Linear(2048, self.modules_size)
            self.fuse_fc = nn.Linear(2048 + self.modules_size, 2048)
            self.getfg_num = 32
            self.generatesc_gru = nn.GRUCell(input_size=2048, hidden_size=2048).cuda()
            self.gru = nn.GRUCell(input_size=2048, hidden_size=2048).cuda()
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
            self.appearance_transform = nn.Linear(15, 2048)
        self.sc_forward_fc1 = nn.Linear(2048, 5)
        self.sc_forward_fc2 = nn.Linear(5, 2048)
        self.u_matrix = nn.Linear(24, 1, bias=False)
        self.u_matrix_add = nn.Linear(12, 24, bias=False)

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

    def edge_box_layer(self,rois, im_info):

        n_boxes = len(rois)  # 128, 256
        # allow boxes to sit over the edge by a small amount
        # _allowed_border =  0
        # map of shape (..., H, W)
        # height, width = rpn_cls_score.shape[1:3]

        # print ">>>>>>>>>>>>>>>>>>>>>>>>union_boxes"
        # print ">>>>>>>>>>>>>>>>>>>>>>>>",len(rois)
        rois = rois.tolist()
        im_info = im_info.tolist()

        union_boxes = []
        im_info = im_info[0]
        # print im_info
        for i in range(n_boxes):
            for j in range(n_boxes):
                # if i == j:
                #     iou = 1.0
                # else:
                #     iou = cal_iou(rois[i], rois[j])

                if True:
                    box = []

                    cx1 = (rois[i][1] + rois[i][3]) * 0.5
                    cy1 = (rois[i][2] + rois[i][4]) * 0.5
                    w1 = (rois[i][3] - rois[i][1]) * 1.0
                    h1 = (rois[i][4] - rois[i][2]) * 1.0

                    if w1 < 0:
                        w1 = 0
                    if h1 < 0:
                        h1 = 0

                    s1 = w1 * h1

                    cx2 = (rois[j][1] + rois[j][3]) * 0.5
                    cy2 = (rois[j][2] + rois[j][4]) * 0.5
                    w2 = (rois[j][3] - rois[j][1]) * 1.0
                    h2 = (rois[j][4] - rois[j][2]) * 1.0

                    if w2 < 0:
                        w2 = 0
                    if h2 < 0:
                        h2 = 0

                    s2 = w2 * h2

                    box.append(w1 / (im_info[0] + 1))
                    box.append(h1 / (im_info[1] + 1))
                    box.append(s1 / ((im_info[0] + 1) * (im_info[1] + 1)))

                    box.append(w2 / (im_info[0] + 1))
                    box.append(h2 / (im_info[1] + 1))
                    box.append(s2 / ((im_info[0] + 1) * (im_info[1] + 1)))

                    box.append((cx1 - cx2) / (w2 + 1))
                    box.append((cy1 - cy2) / (h2 + 1))

                    box.append(pow((cx1 - cx2) / (w2 + 1), 2))
                    box.append(pow((cy1 - cy2) / (h2 + 1), 2))

                    box.append(math.log((w1 + 1) / (w2 + 1)))
                    box.append(math.log((h1 + 1) / (h2 + 1)))

                else:
                    box = [0] * 12
                    # index += 1

                union_boxes.append(box)

        edge_boxes = np.array(union_boxes).astype(np.float32)
        return edge_boxes

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        list110 = [1, 1, 0]
        list000 = [0, 0, 0]
        list010 = [0, 1, 0]
        base_feat = self.RCNN_base(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data
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
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        #base_feat = self.RCNN_base(im_data)
        scale = base_feat.size(2) / im_info[0][0]
        if cfg.POOLING_MODE == 'align':
            roi_pool_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5), scale)
        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        pooled_feat = self._head_to_tail(roi_pool_feat)



        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if bbox_pred.size(0) == 1:
            bbox_pred = bbox_pred[0]

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_weights = list(self.RCNN_cls_score.named_parameters())
        cls_weight = cls_weights[0][1]

        if cls_score.size(0) == 1:
            cls_score = cls_score[0]
        cls_prob = F.softmax(cls_score, 1)
        cls_pro_1st = cls_prob
        cls_index1 = torch.max(cls_prob, 1)[1]
        cls_index1 = cls_index1.unsqueeze(0)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_cls_2nd = 0
        RCNN_loss_bbox_2nd = 0

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
            rois = bbox_decode(rois, bbox_pred, batch_size, self.class_agnostic, self.n_classes, im_info, self.training, cls_prob)
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

            if cfg.POOLING_MODE == 'align':
                ####################################   场景：ranksc_re或ranksc_cat，得到sc_box，然后丢入self._PyramidRoI_Feat，self._head_to_tail_2nd中
                if  self.modules_exist[1] == 2:
                    sc_box = [0, 0, 0, im_info[0][0], im_info[0][1]]
                    sc_box = torch.Tensor(sc_box).cuda()
                    sc_box = torch.unsqueeze(sc_box, dim=0)

                    sc_rois = torch.cat((sc_box, rois), dim=0)  # rois[129,5]
                    roi_pool_feat = self.RCNN_roi_align(base_feat, sc_rois.view(-1, 5), scale)
                    sc_pooled_feat2 = self._head_to_tail_2nd(roi_pool_feat)
                    pooled_feat2 = sc_pooled_feat2[1:, :]
                    spat_roi_pool_feat_1 = self.bbox_conv1(roi_pool_feat)
                    spat_roi_pool_feat_1 = self.relu(spat_roi_pool_feat_1)
                    spat_roi_pool_feat_2 = self.bbox_conv2(spat_roi_pool_feat_1)
                    spat_roi_pool_feat_2 = self.relu(spat_roi_pool_feat_2)
                    spat_roi_pool_feat_avg = self.avgpooling(spat_roi_pool_feat_2)
                    spat_roi_pool_feat_avg_pro = torch.squeeze(spat_roi_pool_feat_avg)
                    
                    
                    
                else:
                    roi_pool_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5), scale)
                    pooled_feat2 = self._head_to_tail_2nd(roi_pool_feat)
                ####################################

            elif cfg.POOLING_MODE == 'pool':
                roi_pool_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
                pooled_feat2 = self._head_to_tail_2nd(roi_pool_feat)
            # cls_pool_feat = roi_pool_feat.flatten(start_dim=1)
            # cls_pool_feat = F.relu(self.fc7(cls_pool_feat))
            # cls_pool_feat = F.relu(self.fc8(cls_pool_feat))
            cls_pooled_feat = pooled_feat2
            bbox_pooled_feat = spat_roi_pool_feat_avg_pro[1:,:]
            pooled_feat_023 = pooled_feat2.clone()
            scence_lable = 0
            if self.modules_exist[1] == 2:
                ##   场景：ranksc_re或ranksc_cat, 抽离出原pooled_feat[128，1024]，得scence_feat[1,1024]
                if  self.sc_net_type_flag[1] == 1:
                    rois = sc_rois[1:, :]  # [128,5]
                    scence_feat = torch.unsqueeze(sc_pooled_feat2[0, :], dim=0)  # [1,1024]
                    cls_pooled_feat = sc_pooled_feat2[1:, :]  # [128,1024]

                    # 使用genratesc_gru对scence_feat进行更新，输入:scence_feat 隐状态:前32个pooled_feature
                    for i in range(self.getfg_num):
                        scence_feat = self.generatesc_gru(torch.unsqueeze(cls_pooled_feat[i], dim=0), scence_feat)
                    # scence_feat_new1 = self.sc_forward_fc1(scence_feat)
                    # scence_feat_new1 = F.softmax(scence_feat_new1)
                    # scence_lable_max = torch.max(scence_feat_new1, 1)
                    # scence_lable = str(scence_lable_max[1].data)
                    # scence_feat_new2 = self.sc_forward_fc2(scence_feat_new1)
                    if self.training:  # 训练时128
                        scence_feat_new2 = scence_feat.repeat(128, 1)
                    else:  # 测试时300，简单改法，后续改为标准型
                        scence_feat_new2 = scence_feat.repeat(300, 1)
                    seq_len = 2
                    h1 = cls_pooled_feat

                    for i in range(seq_len):
                        h1 = self.gru(scence_feat_new2, h1)
                    if self.sc_net_type_flag[0] == 1:  # re型
                        cls_pooled_feat = h1
                    if self.sc_net_type_flag[1] == 1:  # cat型
                        sc_tocat = self.sc_tocat(h1)  # [128,256]
                        cls_pooled_feat = torch.cat((sc_tocat, cls_pooled_feat), -1)  # [128,1024+256]
                        pooled_feat_023 = torch.cat((pooled_feat_023, sc_tocat), -1)
                        # pooled_feat = self.fuse_fc(pooled_feat)  # [128,1024+256]---->[128,1024]
                ###################################

            visual_fea = 2
            if self.hkrms_exist[1] == 1:
                occu_pooled_feat = cls_pooled_feat
                if self.modules_exist[2] == 3:
                    ########################使用GCN处理隐含空间知识，取topk个空间关系##################################
                    # top = len(cls_pro_1st[0])
                    if visual_fea == 2:
                        top = 8
                        edge_boxes = self.edge_box_layer(rois, im_info)
                        n_boxes = len(rois)  # 128
                        ofe = edge_boxes
                        fe = torch.from_numpy(ofe.reshape(n_boxes, n_boxes, 12))  # [128,128,12] 候选框候选框之间的12维关系。
                        fe = Variable(fe.cuda())
                        afe = self.u_matrix_add(fe)
                        PE = self.u_matrix(afe)
                        PE = F.relu(PE.view(n_boxes, n_boxes))
                        nPE = F.normalize(PE, p=2, dim=1)
                        PE_normal = F.softmax(nPE)
                        # PE_normal = nPE

                        appearance_feat_b3 = torch.mm(cls_pro_1st, cls_weight)
                        pooled_feat2 = bbox_pooled_feat.view(batch_size, -1, 2048)
                        Z_feat = (self.feat_change_shape(pooled_feat2))
                        Z_feat_T = torch.transpose(Z_feat, 1, 2)
                        adjs_inter = torch.bmm(Z_feat, Z_feat_T)
                        adjs_inter_view = adjs_inter.view(n_boxes, n_boxes)

                        adjs_inter_normal = F.softmax(adjs_inter_view)
                        Adj_relation_new = adjs_inter_normal
                        Adj_relation = Adj_relation_new

                        for b in range(batch_size):  # 取每个batch单独进行操作
                            # adjs_inter_single = adjs_inter[b, :, :].clone()
                            # adjs_inter_single_new = adjs_inter[b, :, :].clone()
                            adjs_inter_single_new = F.normalize(adjs_inter_normal, p=2, dim=1)
                            adjs_inter_single_new2 = adjs_inter_view.clone()
                            for i in range(len(adjs_inter_single_new2[0])):
                                PE_normal[i] = (PE[i] - PE[i].min()) / (PE[i].max() - PE[i].min())
                                adjs_inter_normal[i] = (adjs_inter_view[i] - adjs_inter_view[i].min()) / (
                                            adjs_inter_view[i].max() - adjs_inter_view[i].min())
                                Adj_relation[i] = PE_normal[i].mul(adjs_inter_normal[i])
                                Adj_relation_new[i] = (Adj_relation[i] - Adj_relation[i].min()) / (
                                        Adj_relation[i].max() - Adj_relation[i].min())
                                adjs_inter_single_new[i] = Adj_relation_new[i]

                                list_adjs, adj_index = torch.sort(adjs_inter_single_new[i], descending=True)  # 取前topK个值
                                one = torch.ones_like(adjs_inter_single_new[i])
                                zero = torch.zeros_like(adjs_inter_single_new[i])
                                adjs_inter_single_new2[i] = torch.where(adjs_inter_single_new[i] > list_adjs[top], one,
                                                                        zero)
                            Adj_s_inter = self.normalization(adjs_inter_single_new2)
                            # visual_feat_b2 = bbox_pooled_feat[b, :, :]
                            transfer_feat_s_inter = self.GCN_inter(appearance_feat_b3, Adj_s_inter)
                            transfer_feat_s_inter = transfer_feat_s_inter.view(batch_size, -1, self.modules_size)

                            # pooled_feat2 = torch.cat((visual_feat2, transfer_feat_s_inter), -1)
                            # spat_pooled_feat = pooled_feat2
                            # pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_s_inter), -1)
                            spat_pooled_feat = torch.cat((pooled_feat2, transfer_feat_s_inter), -1)
                            pooled_feat_023 = torch.cat((pooled_feat_023, transfer_feat_s_inter[0, :, :]), -1)
                            # pooled_feat2 = torch.cat((pooled_feat_sc, transfer_feat_s_inter), -1)


                    if visual_fea == 0:
                        top = 8
                        edge_boxes = self.edge_box_layer(rois, im_info)
                        n_boxes = len(rois)  # 128
                        ofe = edge_boxes
                        fe = torch.from_numpy(ofe.reshape(n_boxes, n_boxes, 12))  # [128,128,12] 候选框候选框之间的12维关系。
                        fe = Variable(fe.cuda())
                        afe = self.u_matrix_add(fe)
                        PE = self.u_matrix(afe)
                        PE = F.relu(PE.view(n_boxes, n_boxes))
                        nPE = F.normalize(PE, p=2, dim=1)
                        PE_normal = F.softmax(nPE,0)
                        adjs_inter_single_new2 = PE_normal.clone()
                        appearance_feat_b3 = torch.mm(cls_pro_1st, cls_weight)



                        for i in range(len(PE_normal[0])):
                            PE_normal[i] = (PE[i] - PE[i].min()) / (PE[i].max() - PE[i].min())




                            list_adjs, adj_index = torch.sort(PE_normal[i], descending=True)  # 取前topK个值
                            one = torch.ones_like(PE_normal[i])
                            zero = torch.zeros_like(PE_normal[i])
                            adjs_inter_single_new2[i] = torch.where(PE_normal[i] > list_adjs[top], one,
                                                                    zero)
                        Adj_s_inter = self.normalization(adjs_inter_single_new2)
                        transfer_feat_s_inter = self.GCN_inter(appearance_feat_b3, Adj_s_inter)
                        transfer_feat_s_inter = transfer_feat_s_inter.view(batch_size, -1, self.modules_size)

                        # pooled_feat2 = torch.cat((visual_feat2, transfer_feat_s_inter), -1)
                        # spat_pooled_feat = pooled_feat2
                        # pooled_feat2 = torch.cat((pooled_feat2, transfer_feat_s_inter), -1)
                        spat_pooled_feat = torch.cat((pooled_feat2, transfer_feat_s_inter[0, :, :]), -1)
                        pooled_feat_023 = torch.cat((pooled_feat_023, transfer_feat_s_inter[0, :, :]), -1)
                        # pooled_feat2 = torch.cat((pooled_feat_sc, transfer_feat_s_inter), -1)






                    ########################使用GAT处理隐含空间知识，##################################




                yinzi2 = int(self.modules_exist[0] > 0) + int(self.modules_exist[1] > 0) + int(
                    self.modules_exist[2] > 0)
                # pooled_feat = pooled_feat2.contiguous().view(-1, 2048 + self.modules_size * yinzi2)

            # compute bbox offset
            # 控制使用哪种模型
            if self.hkrms_exist[0]:#110
                if self.modules_exist[2] == 0  and self.modules_exist[1] == 0:
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat2)
                if (self.modules_exist[2] == 3 and self.modules_exist[1] == 0):  # 003110
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat_023)
                if self.modules_exist[2] == 0 and self.modules_exist[1] == 2:  # 020110
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat_023)
                if self.modules_exist == self.list023:
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat_023)
            else:
                if self.modules_exist[2] == 3:#003
                    # bbox_pooled_feat = self.spat_bbox_fully_connect(spat_pooled_feat)
                    bbox_pred = self.RCNN_bbox_pred_2nd(spat_pooled_feat)
                else:
                    bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat_023)
            if self.modules_exist == list000 and self.hkrms_exist == list010:
                box_pooled_feat = self.bbox_fully_connect(pooled_feat2)
                # box_pooled_feat = F.relu(box_pooled_feat)
                bbox_pred = self.RCNN_bbox_pred_2nd(box_pooled_feat)

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
                if self.modules_exist[2] == 0  and self.modules_exist[1] == 0:
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat2)
                if (self.modules_exist[2] == 3 and self.modules_exist[1] == 0):  # 003110
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat_023)
                if self.modules_exist[2] == 0 and self.modules_exist[1] == 2:  # 020110
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat_023)
                if self.modules_exist == self.list023:
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat_023)
            else:
                if self.modules_exist[1] == 2:
                    # cls_pooled_feat = self.occu_cls_fully_connect(occu_pooled_feat)
                    cls_score = self.RCNN_cls_score_2nd(occu_pooled_feat)
                else:
                    cls_score = self.RCNN_cls_score_2nd(pooled_feat_023)
            if self.modules_exist == list000 and self.hkrms_exist == list010:
                cls_pooled_feat = self.cls_fully_connect(pooled_feat)
                # cls_pooled_feat = F.relu(cls_pooled_feat)
                cls_score = self.RCNN_cls_score_2nd(cls_pooled_feat)

            if cls_score.size(0) == 1:
                cls_score = torch.squeeze(cls_score)
            cls_prob_2nd = F.softmax(cls_score, 1)
            cls_index2 = torch.max(cls_prob_2nd, 1)[1]
            cls_index2 = cls_index2.unsqueeze(0)

            RCNN_loss_cls_2nd = 0
            RCNN_loss_bbox_2nd = 0
            RCNN_loss_cls_3rd = 0
            RCNN_loss_bbox_3rd = 0


            if self.training:
                # loss (cross entropy) for object classification
                RCNN_loss_cls_2nd = F.cross_entropy(cls_score, rois_label)
                # loss (l1-norm) for bounding box regression
                RCNN_loss_bbox_2nd = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)



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
                # cls_prob_3rd_avg = cls_prob_2nd
                cls_prob_3rd_avg = (cls_pro_1st + cls_prob_2nd) / 2
                # bbox_pred_2nd = bbox_pred_1st
        else:
            if self.modules_exist != list000:
                cls_prob_3rd_avg = cls_prob_2nd
            else:
                cls_prob_3rd_avg = cls_prob_2nd
                # bbox_pred_2nd = bbox_pred_1st

        if self.training:
            return rois, cls_prob_3rd_avg, bbox_pred_2nd, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, rois_label

        return rois, cls_prob_3rd_avg, bbox_pred_2nd, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, rois_label