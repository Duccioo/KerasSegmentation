import argparse
import os
import tensorflow as tf
import sys
import glob

from tensorflow import keras
from keras_segmentation.models.unet import mobilenet_unet

#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()

#cartelle contenenti immagini, maschere e checkpoint
parser.add_argument('--img', dest='img_path', type=str, default='train')
parser.add_argument('--masks', dest='masks_path', type=str, default='train')
parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default='train')
parser.add_argument('--text', dest='text_path', type=str, default='')

#indicano il numero di epoche e quanti step fare ad ogni epoca
parser.add_argument('--epoch', dest='epoch', type=int, default='train')
parser.add_argument('--step_epoch', dest='step_epoch', type=int, default='train')

#indica se ripartire da zero ad ogni train o usare i checkpoint già presenti nella cartella indicata sopra
parser.add_argument('--autoresume', action="store_true")

args = parser.parse_args()

#----------------------------------------------#CONFIG:#---------------------------------------------------------#



#----------------------------------------------#Main#---------------------------------------------------------#

if __name__ == '__main__':

	model = mobilenet_unet(n_classes=2 ,  input_height=512, input_width=512  )

	if args.text_path!="":
		model.train(
			text_path=args.text_path,
			checkpoints_path = args.checkpoint_path, #"/content/prova" , 
			epochs=args.epoch,
			steps_per_epoch=args.step_epoch,
			auto_resume_checkpoint=args.autoresume,
			verify_dataset=False,
		)

	else: 
		model.train(
			train_images = args.img_path, #"/content/img_dai",
			train_annotations = args.masks_path, #"/content/maskBW",
			checkpoints_path = args.checkpoint_path, #"/content/prova" , 
			epochs=args.epoch,
			steps_per_epoch=args.step_epoch,
			auto_resume_checkpoint=args.autoresume,
			verify_dataset=True,
		)