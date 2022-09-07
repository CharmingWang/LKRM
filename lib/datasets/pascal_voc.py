from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom
import matplotlib.pyplot as plt
import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
from lib.pyvgtools.vgeval import VGeval
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

#*********************************************************************************************************
import json
import os.path as osp
#*********************************************************************************************************
# <<<< obsolete


class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        # self._classes = ('__background__',  # always index 0
        #                  'aeroplane', 'bicycle', 'bird', 'boat',
        #                  'bottle', 'bus', 'car', 'cat', 'chair',
        #                  'cow', 'diningtable', 'dog', 'horse',
        #                  'motorbike', 'person', 'pottedplant',
        #                  'sheep', 'sofa', 'train', 'tvmonitor')
        self._classes = ('__background__',  # always index 0
                         'pre-twisted suspension clamp', 'bag-type suspension clamp', 'compression-type strain clamp', 'wedge-type strain clamp',
                         'hanging board', 'u-type hanging ring', 'yoke plate', 'parallel groove clamp',
                         'shockproof hammer', 'spacer', 'grading ring', 'shielded ring',
                         'weight', 'adjusting board')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}
        #*********************************************************************************************************
        # *********************************************************************************************************
        # *********************************************************************************************************
        if image_set == 'train':
            image_set = 'train'
        else:
            import json
            import os.path as osp
            from lib.pyvgtools.vg import VG

            self._VG = VG(self._data_path, self._get_ann_file(), align_dir=image_set)
            cat_ids = self._VG.get_cat_ids()
            # cat_ids = pickle.load(open('/home/jiangchenhan/code/faster-rcnn.pytorch-master2/data/vg/val_ids.pkl', 'rb'))

            cats = self._VG.load_cats(cat_ids)
            self._class_to_vg_id = dict(zip(cats, cat_ids))

        # *********************************************************************************************************
        # *********************************************************************************************************
        # *********************************************************************************************************
        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  str(index) + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def voc_evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    # *********************************************************************************************************************
    # *********************************************************************************************************************
    # *********************************************************************************************************************
    def vg_evaluate_detections(self, all_boxes, output_dir, model_dir, epoch):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_vg_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        # if self._image_set.find('test') == -1:
        self._do_detection_eval(res_file.split('/')[-1], output_dir,  model_dir, epoch)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def _write_vg_results_file(self, all_boxes, res_file):
        results = []
        cnt = 0
        for img_ind, index in enumerate(self.image_index):
            print('Collecting {} results ({:d}/{:d})'.format(index, img_ind+1,
                                                             len(self.image_index)))
            image = {'image_id': index}
            objects, cnt = self._vg_results_one_image([all_boxes[cls_ind][img_ind]
                                                       for cls_ind in range(1, len(self.classes))],
                                                      index, cnt)
            image['objects'] = objects
            results.append(image)
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def _vg_results_one_image(self, boxes, index, cnt):
        results = []
        for cls_ind, cls in enumerate(self.classes[1:]):
            dets = boxes[cls_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cls_ind,
                  # 'category_id': self._class_to_vg_id[cls],
                  'x': xs[k],
                  'y': ys[k],
                  'w': ws[k],
                  'h': hs[k],
                  'object_id': cnt + k,
                  'synsets': [cls],
                  'score': scores[k]} for k in range(dets.shape[0])])
            cnt += dets.shape[0]
        return results, cnt

    # def _do_detection_eval(self, res_file, output_dir):
    #     vg_dt = self._VG.load_res(output_dir, res_file)
    #     vg_eval = VGeval(self._VG, vg_dt)
    #     vg_eval.evaluate()
    #     vg_eval.accumulate()
    #     self._print_detection_eval_metrics(vg_eval)
    #     # eval_file = osp.join(output_dir, 'detection_results.pkl')
    #     # with open(eval_file, 'wb') as fid:
    #     #     pickle.dump(vg_eval, fid, pickle.HIGHEST_PROTOCOL)
    #     # print('Wrote VG eval results to: {}'.format(eval_file))
    def _do_detection_eval(self, res_file, output_dir, model_dir, epoch):
        vg_dt = self._VG.load_res(output_dir, res_file)
        vg_eval = VGeval(self._VG, vg_dt)
        vg_eval.evaluate()
        vg_eval.accumulate()
        self._print_detection_eval_metrics(vg_eval)
        # **********PR曲线绘制**********#
        # vg_eval.summarize()
        self._PR_curve(vg_eval, model_dir, epoch)
        # *****************************#

    def _PR_curve(self, vg_eval, model_dir, epoch):

        pr_f = 0
        for i in range(len(vg_eval.eval['precision'][0, 0, :, 0, 2])):
            pr_f += vg_eval.eval['precision'][0, :, i, 0, 2]
        # vg_eval.eval['precision'][0, :, 0, 0, 2]是第一类的P值，[0,:,0,0,2]对应[TxRxKxAxM]
        # T:IOU阈值[0.5,0.55,...,0.95],共10个,每间隔0.05取一个值
        # R:recall阈值(recall的采样点,PR曲线的横轴)，101个值,0~1每间隔0.01取一个值
        # K:表示检测任务中检测的目标类别，想展示第几类就设为多少
        # A:区域范围[无限制,小目标,中目标,大目标]
        # M:每张图最大检测目标个数,[1,10,100]

        pr_f /= len(vg_eval.eval['precision'][0, 0, :, 0, 2])
        np.savetxt('PR_csv/' + model_dir +'_'+ str(epoch) + '_PR_csv', pr_f, delimiter = ',')

        # pr_array2 = vg_eval.eval['precision'][2,:,0,0,2]
        # pr_array3 = vg_eval.eval['precision'][4,:,0,0,2]
        x = np.arange(0.0, 1.01, 0.01)
        plt.figure()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)


        plt.plot(x, pr_f, 'b-', label='IoU=0.5')
        # plt.plot(x, pr_array2, 'c-', label='IoU=0.6')
        # plt.plot(x, pr_array3, 'y-', label='IoU=0.7')
        plt.legend(loc='lower left')
        plt.savefig('./PR_cure/'+ model_dir +'_'+ str(epoch) + '_PR.png')
        plt.show()
        print('!')

    # *********************************#

    def _get_ann_file(self):
        return osp.join(self._data_path, 'objects_' + self._image_set + '.json')

    def _print_detection_eval_metrics(self, vg_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(vg_eval, thr):
            ind = np.where((vg_eval.params.iouThrs > thr - 1e-5) &
                           (vg_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = vg_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(vg_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(vg_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__' or cls == 'pigeon.n.01':
                continue
            # minus 1 because of __background__
            precision = vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{}: {:.1f}'.format(cls, 100 * ap))

        print('~~~~ Summary metrics ~~~~')
        vg_eval.summarize()

# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
