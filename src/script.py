import argparse
from track import *

parser = argparse.ArgumentParser()
parser.add_argument('-mp','--model_path',help='path to model',type=str)
parser.add_argument('-vp','--video_path',help='path to the video',type=str)
parser.add_argument('-od','--output_dir',help='path to save the video',type=str)




if __name__=='__main__':
    args = parser.parse_args()
    out_dir = args.output_dir
    model_path = args.model_path
    video_path = args.video_path
    
    dl = datasets.LoadVideo(video_path, (1088,608))
    opt = opts().init()
    opt.load_model = model_path
    show_image = False
    output_dir = out_dir
    eval_seq(opt, dl, 'mot',save_dir=output_dir, show_image=show_image)
