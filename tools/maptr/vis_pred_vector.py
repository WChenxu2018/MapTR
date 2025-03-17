import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]
# we choose these samples not because it is easy but because it is hard
CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.2, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)


    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()


    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['#FFA500', 'black', '#1E90FF']

    CLASS2LABEL = {
        'road_border': 0,
        'lane_border': 1,
        'lane_center': 2,
        'others': -1
    }

    logger.info('BEGIN vis test dataset samples gt label & pred')

    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue
       
        metas = data['meta'].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        timestamp = metas[0]['timestamp']
        bag_md5 = metas[0]['bag_md5']
        map1_timestamp = metas[0]['map1_timestamp']
        map2_timestamp = metas[0]['map2_timestamp']

        # import pickle
        # with open(f'test.pkl', 'wb') as f:
        #     pickle.dump(data, f)
        
        with torch.no_grad():
            result = model(return_loss=False, model_input_map1 = data['model_input_map1'],
                           model_input_map2 = data['model_input_map2'], 
                           model_input_map1_mask = data['model_input_map1_mask'],
                           model_input_map2_mask = data['model_input_map2_mask'])
        sample_dir = osp.join(args.show_dir, bag_md5, str(timestamp))
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))
        fig, axs = plt.subplots(1, 5, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 4, 4, 4, 4]})

        # 坐标轴设置
        ax = axs[0]
        start, end = -20, 40  # y 轴范围
        ax.plot([0, 0], [start, end], 'k-', lw=2)  # 绘制主刻度线

        # 标记刻度
        for i in range(start, end + 1, 10):
            ax.plot([-0.2, 0.2], [i, i], 'k-')  # 绘制刻度线
            ax.text(-1, i, f'{i}m', ha='right', va='center')  # 绘制刻度标签

        ax.set_xlim(-1, 1)
        ax.axis('off')
        # 输入图像1
        ax = axs[1]
        for line_index in range(0, data['model_input_map1'].shape[1]):
            xs = []
            ys = []
            line_type = -1
            for vector_index in range(0, data['model_input_map1'].shape[2]):
                if data['model_input_map1_mask'][0][line_index][vector_index] == 0:
                    continue
                x_point = data['model_input_map1'][0][line_index][vector_index][0].item()
                y_point = data['model_input_map1'][0][line_index][vector_index][1].item()
                line_type = int(data['model_input_map1'][0][line_index][vector_index][4].item())
                xs.append(x_point)
                ys.append(y_point)
            if len(xs) == 0:
                continue
            xs = np.array(xs)
            ys = np.array(ys)
            ax.plot(xs, ys, color=colors_plt[line_type], linewidth=1, alpha=0.8, zorder=-1)
            ax.scatter(xs, ys, color=colors_plt[line_type], s=10, alpha=0.8, zorder=-1)
        ax.set_title(f'Input Image 1 {timestamp}')
        ax.axis('off')

        # 输入图像2
        ax = axs[2]
        debug_info = []
        for line_index in range(0, data['model_input_map2'].shape[1]):
            xs = []
            ys = []
            for vector_index in range(0, data['model_input_map2'].shape[2]):
                if data['model_input_map2_mask'][0][line_index][vector_index] == 0:
                    continue
                x_point = data['model_input_map2'][0][line_index][vector_index][0].item()
                y_point = data['model_input_map2'][0][line_index][vector_index][1].item()
                line_type = int(data['model_input_map2'][0][line_index][vector_index][4].item())
                xs.append(x_point)
                ys.append(y_point)
            if len(xs) == 0:
                continue
            debug_info.append({
                "type": line_type,
                "xs": xs,
                "ys": ys
            })
            xs = np.array(xs)
            ys = np.array(ys)
            ax.plot(xs, ys, color=colors_plt[line_type], linewidth=1, alpha=0.8, zorder=-1)
            ax.scatter(xs, ys, color=colors_plt[line_type], s=10, alpha=0.8, zorder=-1)
        ax.set_title('Input Image 2')
        ax.axis('off')
        import json
        with open("test.json", 'w') as fout:
            json.dump(debug_info, fout, indent=4)

        ax = axs[3]
        # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
        gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
        for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
            # import pdb;pdb.set_trace() 
            pts = gt_bbox_3d.numpy()
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            ax.scatter(x, y, color=colors_plt[gt_label_3d],s=0.1,alpha=0.8,zorder=-1)
            # plt.plot(x, y, color=colors_plt[gt_label_3d])
            # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
        ax.set_title('Ground Truth')
        ax.axis('off')

        ax = axs[4]
        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        keep = scores_3d > args.score_thresh
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):

            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            # plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            # plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=0.1,alpha=0.8,zorder=-1)
        # plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        ax.set_title('Inference Result')
        ax.axis('off')
        for ax in axs:
            ax.axis('equal')
            ax.set_xlim(-25, 25)  # 根据实际数据范围调整
            ax.set_ylim(-20, 50)  # 
        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        meta_info = f"md5: {bag_md5}, map1_timestamp: {map1_timestamp}, map2_timestamp: {map2_timestamp}"
        fig.text(0.5, 0.95, meta_info, ha='center', va='center', fontsize=12)
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
