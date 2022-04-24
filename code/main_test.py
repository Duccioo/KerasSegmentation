import argparse
import os
import sys
import glob
from keras_segmentation.predict import model_from_checkpoint_path, predict
from IPython.display import Image



#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()

#cartelle contenenti immagini, maschere e checkpoint
parser.add_argument('--img', dest='img_path', type=str, default='train')
parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default='train')

args = parser.parse_args()

#----------------------------------------------#Main#---------------------------------------------------------#

if __name__ == '__main__':
  
    
    out = predict(
        checkpoints_path= args.checkpoint_path,
        inp=args.img_path,
     
        out_fname="/content/out.png"
    )

    Image('/content/out.png')