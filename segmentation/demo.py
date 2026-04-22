import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from argparse import ArgumentParser
from pathlib import Path
import cv2

from model.model import TwinLiteNetPlus
from utils import val, netParams
from loss import TotalLoss
from demoDataset import LoadImages, LoadStreams
from tqdm import tqdm
import time
import numpy as np
import os
import shutil
from src.lane_splitter import split_da_into_lanes, draw_lane_lines, TemporalSmoother, DAMaskAccumulator
from src.road_info import get_road_info_for_splitting



def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    
    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] ==1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    return img

def detect(args):

    device = "cuda:0"
    half = True
    model = TwinLiteNetPlus(args)
    model = model.cuda()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if args.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=args.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)
        bs = 1  # batch_size
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device != 'cpu' else None  # run once

    model.load_state_dict(torch.load(args.weight))
    model.eval()

    lane_smoother = TemporalSmoother(alpha=0.3) if args.num_lanes > 1 else None
    da_accum = DAMaskAccumulator(alpha=0.25, threshold=0.4) if args.num_lanes > 1 else None

    # Query road info from OSM if GPS coordinates provided
    road_info = None
    if args.gps_lat is not None and args.gps_lon is not None:
        print(f"Querying OSM for road info at ({args.gps_lat}, {args.gps_lon})...")
        road_info = get_road_info_for_splitting(args.gps_lat, args.gps_lon)
        if road_info["source"] != "error":
            print(f"  Road: {road_info.get('road_name', 'unnamed')} ({road_info['highway_type']})")
            print(f"  Lanes (ego): {road_info['num_lanes']}, "
                  f"Ego width: {road_info['ego_width_m']:.1f}m, "
                  f"Total width: {road_info['total_width_m']:.1f}m "
                  f"[source: {road_info['source']}]")
        else:
            print(f"  OSM query failed: {road_info.get('error', 'unknown')}, using fallback")
            road_info = None

    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.cuda().half() / 255.0 if half else img.cuda().float() / 255.0


        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]
        # Inference
        # t1 = time_synchronized()
        da_seg_out,ll_seg_out= model(img)
        # t2 = time_synchronized()

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
        

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing


        # Generate synthetic lane dividers from DA mask
        if args.num_lanes > 1:
            lane_polylines = split_da_into_lanes(
                da_seg_mask, args.num_lanes,
                smooth_window=args.smooth_window,
                cutoff_bottom=args.cutoff_bottom,
                lane_ratio_threshold=args.lane_ratio,
                temporal_smoother=lane_smoother,
                road_info=road_info,
                da_accumulator=da_accum)
        else:
            lane_polylines = []

        img_vis = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _)

        # Draw synthetic lane dividers on top
        if lane_polylines:
            # Scale polylines from DA mask space to output image space (1280x720)
            vis_h, vis_w = img_vis.shape[:2]
            da_h, da_w = da_seg_mask.shape
            scale_x = vis_w / da_w
            scale_y = vis_h / da_h
            scaled_polylines = []
            for pl in lane_polylines:
                scaled_polylines.append([(int(x * scale_x), int(y * scale_y)) for x, y in pl])
            draw_lane_lines(img_vis, scaled_polylines,
                            color=args.lane_color, thickness=args.lane_thickness)

        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_vis)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                vis_h, vis_w = img_vis.shape[:2]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (vis_w, vis_h))
            vid_writer.write(img_vis)

        else:
            cv2.imshow('image', img_vis)
            cv2.waitKey(1)  # 1 millisecond





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--num-lanes', type=int, default=2, help='number of lanes in ego direction for synthetic lane splitting')
    parser.add_argument('--cutoff-bottom', type=float, default=0.08, help='fraction of image bottom to ignore (hood/dashboard)')
    parser.add_argument('--smooth-window', type=int, default=51, help='smoothing window size for lane divider lines')
    parser.add_argument('--lane-thickness', type=int, default=4, help='thickness of lane divider lines')
    parser.add_argument('--lane-color', type=int, nargs=3, default=[0, 255, 255], help='BGR color for lane dividers')
    parser.add_argument('--lane-ratio', type=float, default=1.4, help='min road_width/ego_width ratio to allow lane splitting')
    parser.add_argument('--gps-lat', type=float, default=None, help='GPS latitude for OSM road info query')
    parser.add_argument('--gps-lon', type=float, default=None, help='GPS longitude for OSM road info query')
    opt = parser.parse_args()
    opt.lane_color = tuple(opt.lane_color)
    with torch.no_grad():
        detect(opt)