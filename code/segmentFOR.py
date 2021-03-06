#Funzione che dato una cartella di immagini e le relative maschere ricava vari indici (matrice di confusione,dice,jaccard)
#-opzioni:-#
#-c per salvare le versioni delle maschere e dell'output della rete a colori
#-u per saltare le immagini tutte nere
#-n per far svolgere alla rete solo un numero n finito di immagini

from PIL import Image
import glob, os
import numpy as np
from datetime import datetime
from BW import convert_BW
import argparse
import cv2
from sklearn.metrics import confusion_matrix,jaccard_score
from sklearn.metrics import accuracy_score
from keras_segmentation.predict import predict, model_from_checkpoint_path,overlay_seg_image
from scipy.spatial.distance import dice

#opzioni per utilizzare solo n immagini e saltare quelle che ritornano 1.0
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_img", help="", type=int)#-n per far svolgere alla rete solo un numero n finito di immagini
parser.add_argument("-u", "--no1", help="",action="store_true")#-u per saltare le immagini tutte nere
parser.add_argument("-c", "--color", help="",action="store_true") #-c per salvare le versioni delle maschere e dell'output della rete a colori
parser.add_argument("--overlay", help="",action="store_true") 
parser.add_argument('--output', dest='output_path', type=str)
parser.add_argument('--input', dest='input_path', type=str)
parser.add_argument('--log', dest='log_path', type=str)
parser.add_argument('--checkpoint', dest='checkpoint_path', type=str)

args = parser.parse_args()

#CONFIG:
dir_log= args.log_path #cartella per salvare i file di log
dir_in= args.input_path
#cartella per prendere le maschere di test
dir_out=args.output_path #cartella per salvare le maschere della rete
path_model= args.checkpoint_path #percorso dove è il modello della rete già addestrato
#vado a prendere l'ora per salvare il file di log
now = datetime.now()
date = now.strftime("%d_%m-%H_%M")

label_colours = [(0,0,0), (255,255,255)]
                #0=unclassified, 1=globoli

#------------------------------------------------------------------------------#
#variante Jaccard (usando operazioni tra vettori booleani)
def jaccard_binary(x,y):
  x = np.asarray(x, bool) 
  y = np.asarray(y, bool) 
  if np.bitwise_and(x, y).sum() == 0 and np.bitwise_or(x, y).sum() == 0:
        return 1.0
  return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

#Dice
def dice_loss(inputs, target):
    num = np.size(target,0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

def dice_alt(pred, true, k = 255):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice
#Confusion Matrix
def compute_confusion_matrix(inputs,target):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    inputs1=inputs.reshape(-1)
    target1=target.reshape(-1)

    CM = confusion_matrix(target1,inputs1)
    if CM.ndim==1:
      TN=CM[0][0]
    else:
      TN = CM[0][0]
      try:
        FP = CM[0][1]
        FN = CM[1][0]
        TP = CM[1][1]
      except:
        error=1
    return(TP, FP, TN, FN)

#ACCURACY
def ACCURACY(tp,fp,tn,fn):
  n=float(tp+tn+fp+fn)
  if n==0:
    return 0
  else:
   return float((tp+tn)/(tp+tn+fp+fn))
  
#PRECISION
def PRECISION(tp,fp,tn,fn):
  if (tp+fp)==0 and (tp)==0:
    return 1
  elif (tp+fp)==0:
    return 0
  else:
    return float((tp)/(tp+fp))

#RECALL    
def RECALL(tp,fp,tn,fn):
  if (tp+fn)==0 and (tp)==0:
    return 1  
  elif (tp+fn)==0:
    return 0
  else:
    return float((tp)/(tp+fn))

#F1 score:
def F1_SCORE(tp,fn,fp):
  if tp==0 and tp+( (fn+fp)/2 )==0:
    return 1
  elif tp+( (fn+fp)/2 )==0:
    return 0
  else:
    return tp/( tp+( (fn+fp)/2 ) )


#funzione per creare immagine che evidenzia glomeruli con colori
def imshow_components(labels,path):
    # Map component labels to hue val
    if np.max(labels)==0:
        label_hue = np.uint8((179*labels)/1)
    else:
      label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0
    cv2.imwrite(path, labeled_img)
  
#-------------------------------------MAIN---------------------------------------------#

i=0
model_in= model_from_checkpoint_path(path_model)
MediaJaccard=0

os.chdir(dir_in) #cartella dove sono le immagini
print("----------------Start---------------")
for file in glob.glob("*.jpg"): #ciclo le immagini dentro la cartella

  if args.n_img and i==args.n_img: #controllo parametro opzionale n
    break  

  in_mask_path=os.path.join(dir_in,file.replace(".jpg","")+'_seg.png') #path immagine di test
  
  target_img = Image.open(in_mask_path) #prendo l'immagine già segmentata di test
  
  img = file
  out_img_path=os.path.join(dir_out,"OUT_"+file) #path dell'immagine che restituisce la rete
  #in_mask_path = dir_out+"IN_"+file #(opzionale) path per salvare l'immagine in Bianco e nero della maschera 
  
  
  y_out =  predict(
    model= model_in,
    inp=file,
    out_fname=out_img_path,
    colors= label_colours,
    overlay_img=False
  )
  
  
        
    
  in_mask= np.array(convert_BW(target_img)) # converto in array la maschera di test
  #decoded_out = np.array(Image.open(out_img_path))
  #decoded_out = y_out
 
  

  #mi calcolo gli indici che mi servono
  #DICE=1-dice(y_out,in_mask)
  DICE=1-dice(in_mask.reshape(-1)/255,y_out.reshape(-1)/255)
  if np.isnan(DICE):
    DICE=0 
  #JACCARDB=jaccard_binary(in_mask/255, y_out/255)
  JACCARDB=jaccard_score(in_mask.reshape(-1)/255,y_out.reshape(-1)/255)
  #(opzionale) posso scartare le immagini che ritornano 1.0 con DICE, ovvero le immagini completamente nere, selezionando no1
  if args.no1 and JACCARDB==1.0: #controllo parametro opzionale no1
    continue
  
  tp, fp, tn, fn=compute_confusion_matrix(in_mask, y_out)
  ACCURACY1=accuracy_score(in_mask.reshape(-1),y_out.reshape(-1))#(tp+tn)/(tp+tn+fp+fn)
  PRESCISION1=PRECISION(tp,fp,tn,fn)
  RECALL1=RECALL(tp,fp,tn,fn)
  F1_SCORE1=F1_SCORE(tp,fn,fp)

  if tn==786432 and fp==0 and tp==0 and fn==0:
    JACCARDB=1
    DICE=1
  #preparo le immagini per contare i glomeruli
  #immagini predette dalla rete
  out_component = cv2.imread(out_img_path, 0)
  out_component = cv2.threshold(out_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
  num_labels_out, labels_out = cv2.connectedComponents(out_component)
  
  #immagini di test
  in_component = cv2.imread(in_mask_path, 0)
  in_component = cv2.threshold(in_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
  num_labels_in, labels_in = cv2.connectedComponents(in_component)

  #stampo gli indici:
  print(str(i+1)+" "+file+":")
  print("Jaccard:",JACCARDB)
  print("Dice: ",DICE)
  print('Matrice di Confusione:','\n [', tn, fp,'] \n [', fn, tp,"]")
  print("Accuracy=",ACCURACY1, " Precision=", PRESCISION1," Recall=",RECALL1)
  print("Numero Glomeruli: rete OUT->",num_labels_out-1,"test IN->",num_labels_in-1)
  
  #opzionale se attivo salva la fusione tra la maschera creata dalla rete e l'immagine originale
  if args.overlay:
    output_img = Image.fromarray(overlay_seg_image(np.array(Image.open(file)),y_out)) 
    output_img.save(out_img_path)#salvo la maschera

  if args.color: #controllo parametro opzionale color se è definito allora:
    #salvo le immagini che evidenziano i glomeruli con colori diversi
    imshow_components(labels_out,out_img_path)

  #calcolo la media
  MediaJaccard=MediaJaccard+JACCARDB

  #creo un stringa da salvare poi in un file di log
  string=file+" "+str(JACCARDB)+" "+str(DICE)+" "+str(num_labels_out-1) +" "+str(num_labels_in-1)+" "+str(ACCURACY1)+" "+str(PRESCISION1)+" "+str(RECALL1)
  f=open(os.path.join(dir_log,"log"+date+".txt"), "a+")
  f.write(string+"\n")
  print("-----------------------------")
  i=i+1
  f.close()      


print('~~~~~~Test Finito~~~~~~ ')
MediaJaccard = MediaJaccard / (i)
print("Media Jaccard:",MediaJaccard)
