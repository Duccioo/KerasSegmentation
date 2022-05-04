
from PIL import Image
import numpy as np
from datetime import datetime
import argparse
import cv2
from sklearn.metrics import confusion_matrix,jaccard_score
from sklearn.metrics import accuracy_score
import BW
from tqdm import tqdm
import mmap
import notifica_telegram
from keras_segmentation.predict import predict, model_from_checkpoint_path,overlay_seg_image
from scipy.spatial.distance import dice

#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_img", help="", type=int)#-n per far svolgere alla rete solo un numero n finito di immagini
parser.add_argument("-u", "--no1", help="",action="store_true")#-u per saltare le immagini tutte nere
parser.add_argument("-c", "--color", help="",action="store_true") #-c per salvare le versioni delle maschere e dell'output della rete a colori
parser.add_argument("--overlay", help="",action="store_true") 
parser.add_argument('--input', dest='input_path', type=str)
parser.add_argument('--output', dest='output_path', type=str, default= "")
parser.add_argument('--log', dest='log_path', type=str)
parser.add_argument('--checkpoint', dest='checkpoint_path', type=str)
parser.add_argument('--fold', dest='fold_path', type=str)
args = parser.parse_args()

#----------------------------------------------#CONFIG:#---------------------------------------------------------#
UNO=True #se è False allora salterà l'output di tutte le immagini nere
COLOR=args.color #se è False allora non salva le immagini colorate
CONTINUE=True #se è True allora se il programma si ferma continuerà da dove si è fermato
dir_fold=args.fold_path #esempio '/content/deeplearn/F1
dir_log=dir_fold+"/LOG/" #cartella per salvare i file di log
dir_txt=dir_fold+"/test.txt" #file dove prendere i nomi delle immagini
dir_in=args.input_path #directory dove prendere le immagini
if args.output_path!="":
  dir_out=args.output_path
else:
  dir_out=dir_fold+"/OUTPUT/" #cartella per salvare le maschere della rete

path_model=dir_fold+"/model/checkpoint" #percorso dove è il modello della rete già addestrato


#identifico il fold
dir_fold_name=dir_fold.split("/")
for part in dir_fold_name:
        if part=="F1":
            dir_fold_name="F1"
            break
        elif part=="F2":
            dir_fold_name="F2"
            break
        elif part=="F3":
            dir_fold_name="F3"
            break   
        elif part=="F4":
            dir_fold_name="F4"
            break
        elif part=="F5":
            dir_fold_name="F5"
            break  
        

#vado a prendere l'ora per salvare il file di log
now = datetime.now()
date = now.strftime("%H_%M")


label_colours = [(0,0,0), (255,255,255)]
                #0=unclassified, 1=globoli

def decode_labels(mask, num_images=1, num_classes=2):
    n, h, w, c = mask.shape
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

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
        TP=0 
      else:
        True
      
     
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

#funzione che ritorna il numero di linee di un file di testo
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines     
#----------------------MAIN---------------------------------#


i=0
h=0
n=0
try:
  f_continue=open(dir_log+"log_"+dir_fold_name+".txt", "x")
except:
  f_continue=open(dir_log+"log_"+dir_fold_name+".txt", "r")
  f_continue_list = f_continue.readlines()
  f_continue.close()
  for string in f_continue_list:
    n=n+1 #n equivale al numero di righe nel file continue

#configurazione media
Media_dice=0
Media_jaccard=0
Media_Nglomeruli=0
Media_accuracy=0
Media_recall=0
Media_F1=0
Media_precision=0

#Apro il file contenente la media e imposto i valori presenti nel file come i valori di media dello step precedente:
try:
  f_media=open(dir_log+"media_"+dir_fold_name+".txt",'x')
except:
  f_media=open(dir_log+"media_"+dir_fold_name+".txt",'r')
  with f_media as f:
    medie_line = f.readlines()
    for media_single in medie_line:
      media=media_single.split("=")
      print(media[0])
      print(media[1])
      if media[0]=="Dice":
        Media_dice=float(media[1].replace("\n",""))*(n) #trovo la sommatoria di tutti i dice 
      elif media[0]=="Jaccard":
        Media_jaccard=float(media[1].replace("\n",""))*(n)
      elif media[0]=="Accuracy":
        Media_accuracy=float(media[1].replace("\n",""))*(n)
      elif media[0]=="Differenza glomeruli":
        Media_Nglomeruli=float(media[1].replace("\n",""))*(n)  
      elif media[0]=="Recall":
        Media_recall =float(media[1].replace("\n",""))*(n) 
      elif media[0]=="Precision":
        Media_precision =float(media[1].replace("\n",""))*(n) 
      elif media[0]=="F1 Score":
        Media_F1 =float(media[1].replace("\n",""))*(n) 
    f.close()



model_in= model_from_checkpoint_path(path_model)   
print("----------------Start---------------")

with open(dir_txt) as textfile:
  totale=get_num_lines(dir_txt)
  
  for line in tqdm(textfile, total=totale): #ciclo le immagini dentro la cartella

    if h!=n and CONTINUE==True: #se h non è uguale a n allora aggiunge 1 ad h e passa al ciclo successivo
      h=h+1 #h è il numero dei passi già percorso, rimane fisso
      i=h#i è il numero dei passi già percorso che scorre a sua volta
      continue
      

    path=line.split(" ")
    path[1]=path[1].replace("/masks/",'')
    in_mask_path=dir_in+"/"+path[1].replace("\n","") #path immagine di test
    target_img = Image.open(in_mask_path) #prendo l'immagine già segmentata di test
    in_img_path=dir_in+"/"+path[0].replace('img/','')
    file=path[0].replace("/img/","") #nome dell'immagine di test
    
    out_img_path= dir_out+"OUT_"+file #path dell'immagine che restituisce la rete
    #in_mask_path = dir_out+"IN_"+file #(opzionale) path per salvare l'immagine in Bianco e nero della maschera 
    
    
    y_out =  predict(
      model= model_in,
      inp=in_img_path,
      out_fname=out_img_path,
      colors= label_colours,
      overlay_img=False
    )
    
    in_mask= np.array(BW.convert_BW(target_img))  # converto in array la maschera di test e la converto in bianco e nero
    
   
    tp, fp, tn, fn=compute_confusion_matrix(in_mask, y_out)
    ACCURACY1=accuracy_score(in_mask.reshape(-1),y_out.reshape(-1))#(tp+tn)/(tp+tn+fp+fn)
    PRESCISION1=PRECISION(tp,fp,tn,fn)
    RECALL1=RECALL(tp,fp,tn,fn)
    F1_SCORE1=F1_SCORE(tp,fn,fp)

    if tn==786432 and fp==0 and tp==0 and fn==0:
      JACCARDB=1
      DICE=1

    else:
      DICE=1-dice(in_mask.reshape(-1)/255,y_out.reshape(-1)/255)
      if np.isnan(DICE):
        DICE=0 
      #JACCARDB=jaccard_binary(in_mask/255, y_out/255)
      JACCARDB=jaccard_score(in_mask.reshape(-1)/255,y_out.reshape(-1)/255)
    
    if UNO==False and JACCARDB==1.0: #controllo parametro opzionale no1
      continue


    #preparo le immagini per contare i glomeruli:
      #immagini predette dalla rete
    out_component = cv2.imread(out_img_path, 0)
    out_component = cv2.threshold(out_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels_out, labels_out = cv2.connectedComponents(out_component)
    
      #immagini di test
    in_component = cv2.imread(in_mask_path, 0)
    in_component = cv2.threshold(in_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels_in, labels_in = cv2.connectedComponents(in_component)
  

    if args.color:
    #salvo le immagini che evidenziano i glomeruli con colori diversi
      imshow_components(labels_out,out_img_path)

    if args.overlay:
      output_img = Image.fromarray(overlay_seg_image(np.array(Image.open(in_img_path)),y_out)) 
      output_img.save(out_img_path)#salvo la maschera

    #preparo la media per salvarla nel file
    Media_dice=Media_dice+DICE
    Media_jaccard=Media_jaccard+JACCARDB
    Media_Nglomeruli=Media_Nglomeruli+((num_labels_out-1)-(num_labels_in-1))
    Media_accuracy=Media_accuracy+ACCURACY1
    Media_F1=Media_F1+F1_SCORE1
    Media_precision=Media_precision+ PRESCISION1
    Media_recall=Media_recall+RECALL1
    
    #creo un stringa da salvare poi in un file di log
    string=file+" "+\
      str(round(JACCARDB,3))+" "+ \
      str(round(DICE,3))+" "+\
      str(num_labels_out-1)+" "+\
      str(num_labels_in-1)+" "+\
      str(round(F1_SCORE1,3))+" "+\
      str(round(ACCURACY1,3))+" "+\
      str(round(PRECISION(tp,fp,tn,fn),3))+" "+\
      str(round(RECALL(tp,fp,tn,fn),3))+" "+\
      str(tp)+" "+str(tn)+" "+str(fp)+" "+str(fn) 
    
    f=open(dir_log+"log_"+dir_fold_name+".txt", "a+")
    f.write(string+"\n")
    f.close()
    
    if i%300==0 or i==0 or ((i+1)/totale)==1:
      notifica_telegram.invio("TEST",dir_fold,i,totale,185857885,str(PRESCISION1)) #-1001541977258 )
    #print("-----------------------------")

    #Salvo le medie dei risultati in un file del drive
    f1=open(dir_log+"media_"+dir_fold_name+".txt", "w")
    f1.write("Dice="+str(Media_dice/(i+1))+'\n') 
    f1.write("Jaccard="+str(Media_jaccard/(i+1))+'\n')  
    f1.write("Differenza glomeruli="+str(Media_Nglomeruli/(i+1))+'\n') 
    f1.write("Accuracy="+str(Media_accuracy/(i+1))+'\n')
    f1.write("Recall="+str(Media_recall/(i+1))+'\n')
    f1.write("Precision="+str(Media_precision/(i+1))+'\n')
    f1.write("F1 Score="+str(Media_F1/(i+1))+'\n')
    f1.write("Images left="+str(totale-i)+"\n")
    f1.write("Last image capture="+file+"\n")
    f1.close()  

    i=i+1

              


print("///////END//////////")
print("Media:")
print("dice=",Media_dice/i) 
print("jaccard=",Media_jaccard/i)  
print("differenza glomeruli=",Media_Nglomeruli/i)  
print("accuracy=",Media_accuracy/i)     
f.close()
    




