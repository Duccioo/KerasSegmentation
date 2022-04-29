from turtle import color
from unicodedata import name
from PIL import Image
import glob, os
import argparse
import tqdm
import time


#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--img', dest='img_path', type=str, default="/")
parser.add_argument('--output', dest='collage_out', type=str, default="")

parser.add_argument('--format', dest='format', type=str, default="jpg")

args = parser.parse_args()


#----------------------------------------------#CONFIG:#---------------------------------------------------------#
def collage_maker( n_image ,path_imgs,  max_x, max_y, min_x,min_y,format="jpg",path_collage_out="NO"):
    n_image=str(n_image)
    plus=""
    #dimensioni immagini
    dim_x=512 
    dim_y=512

    #calcolo la dimensione massima del collage in pixel
    dim_collage_x=(max_x-min_x+1)*dim_x
    dim_collage_y=(max_y-min_y+1)*dim_y

    #creo l'immagine di base del collage tutta nera
    collage = Image.new("RGB",(dim_collage_x, dim_collage_y), color=(0,0,0))

    os.chdir(path_imgs)

    
    for file in tqdm.tqdm(glob.glob("*"+n_image+"_tile*."+format),colour='green',desc=n_image):
        name_file=file.replace("."+format,"")
        name_file=name_file.partition("tile")[2]
        if format=="png":
            name_file=name_file.replace("_seg","")
            plus='_seg'
        
        if file.find("OUT")!=-1:
            plus="_OUT"


        image_x=int(name_file.partition("x")[0])-min_x
        image_y=int(name_file.partition("x")[2])-min_y
        
        
        img = Image.open(file)
        collage.paste(img,(image_x*dim_x,image_y*dim_y))
        
    if path_collage_out!="NO":
        collage.save(path_collage_out+n_image+"_complete"+plus+".jpg")
    
    else: 
        collage.show()

#----------------------------------------------#Main#---------------------------------------------------------#

set_name=set()
set_x=set()
set_y=set()
os.chdir(args.img_path)
for file in glob.glob("*_tile*."+args.format):
    f_name=file.partition("tile")[0].replace( "_","")

    if f_name.find("OUT")!=-1:
        f_name=f_name.replace("OUT","")

    set_name.add(int(f_name))

  

for item in set_name:
     
    for file in glob.glob("*"+str(item)+"_tile*."+args.format):
        f_x=file.partition("tile")[2].partition(".")[0].partition("x")[0]
        f_y=file.partition("tile")[2].partition(".")[0].partition("x")[2]
        
        if file.find("seg")!=-1:
            f_y=f_y.strip("_seg")
        set_x.add(int(f_x))
        set_y.add(int(f_y))    
    
    min_x=min(set_x)
    min_y=min(set_y)   
    max_x=max(set_x)
    max_y=max(set_y)
    collage_maker(item,args.img_path,max_x,max_y,min_x,min_y,path_collage_out=args.collage_out,format=args.format)
    set_x.clear()
    set_y.clear()






