from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        # basic experiment setting
        self.task = 'mot'
        self.dataset = 'jde'
        self.exp_id = 'default'
        self.test = True
        #self.parser.add_argument('--load_model', default='../models/ctdet_coco_dla_2x.pth',
                                 #help='path to pretrained model')
        self.load_model = ''
        self.resume = True

        # system
        self.gpus = '0, 1'
        self.num_workers = 8
        self.not_cuda_benchmark = True
        self.seed = 317
        # log
        self.print_iter = 0
        self.hide_data_time = True
        self.save_all = True
        self.metric = 'loss'
        self.vis_thresh = 0.5
        # model
        self.arch = 'dla_34'
        self.head_conv = -1
        self.down_ratio = 4
        # input
        self.input_res = -1
        self.input_h = -1
        self.input_w = -1
        # train
        self.lr = 1e-4
        self.lr_step = '20,27'
        self.num_epochs = 30
        self.batch_size = 12
        self.master_batch_size = -1
        self.num_iters = -1
        self.val_intervals = 5
        self.trainval = True
        # test
        self.K = 128
        self.not_prefetch_test = True
        self.fix_res = True
        self.keep_res = True
        # tracking
        self.test_mot16 = False
        self.val_mot15 = False
        self.test_mot15 = False
        self.val_mot16 = False
        self.test_mot17 = False
        self.val_mot17 = False
        self.val_mot20 = False
        self.test_mot20 = False
        self.conf_thres = 0.6
        self.det_thres = 0.3
        self.nms_thres = 0.4
        self.track_buffer = 30
        self.min_box_area = 200
        self.input_video = '../videos/MOT16-03.mp4'
        self.output_format = 'video'
        self.output_root = '../results'

        # mot
        self.data_cfg = '../src/lib/cfg/data.json'
        self.data_dir = '../../data/yfzhang/MOT/JDE'

        # loss
        self.mse_loss = True

        self.reg_loss = 'l1'
        self.hm_weight = 1
        self.off_weight = 1
        self.wh_weight = 0.1
        self.id_loss = 'ce'
        self.id_weight = 1.0
        self.reid_dim = 512

        self.norm_wh = True
        self.dense_wh = True
        self.cat_spec_wh = True
        self.not_reg_offset = False

    def parse(self, args=''):
        opt = self
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1: # init default head_conv
          opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
          opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
          opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
          slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
          if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
          opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname('__file__'), '..', '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
          model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                      else opt.save_dir
          opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
          opt.heads = {'hm': opt.num_classes,
                       'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
                       'id': opt.reid_dim}
          if opt.reg_offset:
            opt.heads.update({'reg': 2})
          opt.nID = dataset.nID
          opt.img_size = (1088, 608)
        else:
          assert 0, 'task not defined!'
        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        default_dataset_info = {
          'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
        }
        class Struct:
          def __init__(self, entries):
            for k, v in entries.items():
              self.__setattr__(k, v)
        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
