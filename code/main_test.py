import argparse
from keras_segmentation.predict import predict
import numpy as np



#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()

#cartelle contenenti immagini, maschere e checkpoint
parser.add_argument('--img', dest='img_path', type=str)
parser.add_argument('--checkpoint', dest='checkpoint_path', type=str)

args = parser.parse_args()

#----------------------------------------------#Main#---------------------------------------------------------#

if __name__ == '__main__':

    out = predict(
        checkpoints_path= args.checkpoint_path,
        inp=args.img_path,
        out_fname="/content/out.png"
    )
    print(out.shape)

   
