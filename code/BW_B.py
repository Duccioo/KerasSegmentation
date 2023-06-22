#definita la cartella da riga di comando
#prende tutte le immagini dentro la cartella specificata e le trasforma in bianco e nero nel path specificato
import sys
from PIL import Image
import glob, os


#funzione per convertire in bianco e nero le immagini delle maschere
def convert_BW(img, path=''):
    img = img.convert("RGB")
    datas = img.getdata()
    new_image_data = []
    for item in datas:
        if (item[0]==0 and item[1]==0 and item[2]==255):
          new_image_data.append((0, 0, 0))
        
        
        # ##
        # elif (item[0]==255 and item[1]==255 and item[2]==255):
        #   new_image_data.append((1, 1, 1))

        else:
          new_image_data.append(item)

    img.putdata(new_image_data)
    print(path)
    #(opzionale) salvo l'immagine convertita in bianco e nero
    if path!='':
      img.save(path)
    return img


# folder= sys.argv[1]
# for file in glob.glob(folder+"\*"):
#   # print(file)
#   # convert_BW( Image.open(file),file.replace("_seg",""))
#   convert_BW( Image.open(file),file.replace(".jpg","_BW.jpg"))
#   os.remove(file)