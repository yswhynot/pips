import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

random.seed(125)
np.random.seed(125)

def run_model(model, rgbs, N, sw):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    # H, W = H_, W_
    rgbs_ = rgbs_.reshape(B, S, C, H_, W_)

    # try to pick a point on the dog, so we get an interesting trajectory
    # this is for the three_anchor_down.mp4
    pts = np.array([[440.0, 265.0, 90.0], [190.0, 155.0, 125.0]], dtype=float)
    x = torch.tensor([pts[0, :].tolist()], device="cuda")
    y = torch.tensor([pts[1, :].tolist()], device="cuda")
    xy0 = torch.stack([x, y], dim=-1) # B, N, 2
    _, S, C, H_, W_ = rgbs_.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:,0] = xy0[:,n] # B, 1, 2  # set first position 
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs_[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    # scale the trajs_e to the original size
    trajs_e = trajs_e * torch.tensor([W/W_, H/H_], dtype=torch.float32, device='cuda')
    prep_rgbs = utils.improc.preprocess_color(rgbs)
    #gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    # prep_rgbs = rgbs
    
    if sw is not None and sw.save_this:
        linewidth = 2

        kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1,:, :], prep_rgbs[0:1,:S], cmap='spring', linewidth=linewidth)

        # write to disk, in case that's more convenient
        kp_list = list(kp_vis.unbind(1))
        kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
        kp_list = [Image.fromarray(kp) for kp in kp_list]
        # save as mp4, not gif
        out_fn = './chain_out_%d.mp4' % sw.global_step
        imageio.mimwrite(out_fn, kp_list, fps=10)
        print('saved %s' % out_fn)
            
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')
        
    return trajs_e
    
def main():

    # the idea in this file is to chain together pips from a long sequence, and return some visualizations
    
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 150
    N = 3 # number of points to track

    filenames = glob.glob('./test/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    print("len of files:", len(filenames))
    S = len(filenames)
    max_iters = len(filenames)//(S//2)-1 # run slightly overlapping subseqs

    log_freq = 1 # when to produce visualizations 
    
    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()
    
    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=10,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        try:
            rgbs = []
            for s in range(S):
                fn = filenames[(global_step-1)*S//2+s]
                if s==0:
                    print('start frame', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2,0,1))
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W

            read_time = time.time()-read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                trajs_e = run_model(model, rgbs, N, sw_t)

            iter_time = time.time()-iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)
            
    writer_t.close()

if __name__ == '__main__':
    main()
