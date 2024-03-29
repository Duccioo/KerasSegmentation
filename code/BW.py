#definita la cartella da riga di comando
#prende tutte le immagini dentro la cartella specificata e le trasforma in bianco e nero nel path specificato
import sys

from PIL import Image
import glob, os


#funzione per convertire in bianco e nero le immagini delle maschere
def convert_BW(img, path='', save=False):
    img = img.convert("RGB")
    datas = img.getdata()
    new_image_data = []
    
    # thresh = 10
    # fn = lambda x : 255 if x > thresh else 0
    # img = img.convert('L').point(fn, mode='1')
    for item in datas:
        if (item[0]==0 and item[1]==0 and item[2]==255):
          new_image_data.append((0, 0, 0))
        
        

        else:
          new_image_data.append(item)
        
       
        

        
       

    img.putdata(new_image_data)
    #(opzionale) salvo l'immagine convertita in bianco e nero
    if save==True:
      img.save(path)
    return img


# folder= sys.argv[1]
# convert_BW( Image.open(folder),folder.replace("seg","BW"),True)
folder= sys.argv[1]
for file in glob.glob(folder+"\*"):
  # print(file)
  # convert_BW( Image.open(file),file.replace("_seg",""), True)
  convert_BW( Image.open(file),file.replace(".png","_BW.png"), True)
  os.remove(file)