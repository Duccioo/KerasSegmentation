import argparse
import os
import tensorflow as tf
from model import Model
import sys
import glob

from tensorflow import keras
from keras_segmentation.models.unet import mobilenet_unet





#definisco da linea di comando a quale fold mi sto riferendo
#esemprio fold='/deeplearn/F1/'
parser = argparse.ArgumentParser()
parser.add_argument('--folder', dest='folder', type=str, default='train')

#indica il nomero al quale salvare il checkpoint
parser.add_argument("-c",'--checkpoint', dest='checkpoint', type=int, default='train')

#indica se voglio ricevere una notifica su telegram ogni qual volta faccio il 25% del training
parser.add_argument("-T",'--telegram', action="store_true")

#indica la cartella dove sono le immagini preprocessate
parser.add_argument('--data_preprocessed', dest='data_preprocessed', type=str, default='train')
args = parser.parse_args()
fold=args.folder



last_checkpoint=0

#controllo se nella cartella model del folder è già presente un checkpoint:
model_dir=glob.glob(fold+"/model/*.index")
if model_dir != []:
	#se è presente almeno un file va a prendere quello con il numero maggiore
	#il formato del nome del modello checkpoint: 'model.ckpt-'+'numero' es: model.ckpt-2060.data-00000-of-00001
	sorted_model_dir=sorted(model_dir,key=lambda x: int(x.split('-')[1].replace(".index","")),reverse=True)
	checkpoint=sorted_model_dir[0].replace(".index","")
	try:
		checkpoint2=sorted_model_dir[1].replace(".index","")
	except:
		checkpoint2="error"


	last_checkpoint=int(checkpoint.split('-')[1])
	print(last_checkpoint)
	if last_checkpoint==0:
		checkpoint='./PretrainResnet/resnet_v1_101.ckpt'
else:
	#altrimenti se non c'è nessun checkpoint vecchio allora prende quello di default
	checkpoint='./PretrainResnet/resnet_v1_101.ckpt'

print("CHECKPOINT:",checkpoint)
#per ottimizzare lo spazio occupato sul drive rimuovo i file più vecchi rispetto al checkpoint più nuovo
for i in glob.glob(fold+"/model/*"):
	if i!=(checkpoint+".data-00000-of-00001") and i!=(checkpoint+".index") and i!=(checkpoint+".meta") and i!=(checkpoint2+".data-00000-of-00001") and i!=(checkpoint2+".index") and i!=(checkpoint2+".meta") and i!="checkpoint":
		os.remove(i)
		print("rimozione file",i)
	

#conto il numero di righe nel file
train=fold+"/train.txt"
file = open(train, "r")
nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
n_righe = len(nonempty_lines)

#----------------------------------------------#CONFIG:#---------------------------------------------------------#
split_data= n_righe #numero di immagini di train totali
epoch=10 #numero di epoche
power=0.99 #potenza
batch_size=2
num_steps=(int(split_data/batch_size)*epoch)
save_inverval=args.checkpoint
data_dir=args.data_preprocessed
if args.telegram:
	telegram_chat="-1001541977258"
else:
	telegram_chat="none"



if __name__ == '__main__':
	# Choose which gpu or cpu to use
	#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
	model = mobilenet_unet(n_classes=2 ,  input_height=512, input_width=512  )

	model.train(
		train_images =  "/content/img_dai",
		train_annotations = "/content/maskBW",
		checkpoints_path = "/content/prova" , 
		epochs=3,
		steps_per_epoch=100,
		auto_resume_checkpoint=False,
		verify_dataset=False,
	)