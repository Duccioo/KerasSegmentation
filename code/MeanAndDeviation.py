from PIL import Image
import glob, os
import argparse
import tqdm
#----------------------------------------------#CLI:#---------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--no1", help="",action="store_true")#-u per saltare le immagini tutte nere
parser.add_argument('--text', dest='text_path', type=str)
args = parser.parse_args()

#----------------------------------------------#CONFIG:#---------------------------------------------------------#
#line  ['Nome', 'Jaccard', 'Dice',"# Glomeruli Output","# Glomeruli Input","F1 SCORE","Accuracy","Precision","Recall","TRUE Positive","TRUE Negative","FALSE Positive","FALSE Negative"]
def mean(file_path):
    num_lines, mean_jaccard_sum,mean_jaccard_square=0,0,0
    mean_dice_sum,mean_dice_square=0,0
    mean_glomeruli_sum, mean_glomeruli_square = 0,0
    mean_accuracy_sum, mean_accuracy_square=0,0
    mean_recall_sum, mean_recall_square=0,0
    mean_precision_sum, mean_precision_square=0,0
    mean_f1_sum, mean_f1_square=0,0


    with open(file_path) as textfile:
        for line in textfile: #ciclo le immagini dentro la cartella
            mean_jaccard_sum+=float(line.split(" ")[1]) #prendo il valore di jaccard dal file text
            mean_jaccard_square+=float(line.split(" ")[1])*float(line.split(" ")[2])

            mean_dice_sum+=float(line.split(" ")[2]) #prendo il valore di jaccard dal file text
            mean_dice_square+=float(line.split(" ")[2])*float(line.split(" ")[2])

            mean_glomeruli_sum+=float(line.split(" ")[3]) - float(line.split(" ")[4])#prendo il valore di jaccard dal file text
            mean_glomeruli_square+=(float(line.split(" ")[3]) - float(line.split(" ")[4]))*(float(line.split(" ")[3]) - float(line.split(" ")[4]))

            mean_accuracy_sum+=float(line.split(" ")[6]) #prendo il valore di jaccard dal file text
            mean_accuracy_square+=float(line.split(" ")[6])**2

            mean_recall_sum+=float(line.split(" ")[8]) #prendo il valore di jaccard dal file text
            mean_recall_square+=float(line.split(" ")[8])*float(line.split(" ")[8])

            mean_precision_sum+=float(line.split(" ")[7]) #prendo il valore di jaccard dal file text
            mean_precision_square+=float(line.split(" ")[7])*float(line.split(" ")[7])

            mean_f1_sum+=float(line.split(" ")[5]) #prendo il valore di jaccard dal file text
            mean_f1_square+=float(line.split(" ")[5])*float(line.split(" ")[5])
            num_lines+=1
        
    Media_jaccard=round(mean_jaccard_sum/num_lines,3)
    STD_jaccard=round( (mean_jaccard_square / num_lines - Media_jaccard ** 2) ** 0.5,3)

    Media_dice=round(mean_dice_sum/num_lines,3)
    STD_dice = round((mean_dice_square / num_lines - Media_dice ** 2) ** 0.5,3)

    Media_Nglomeruli=round(mean_glomeruli_sum/num_lines,3)
    STD_Nglomeruli=round((mean_glomeruli_square / num_lines - Media_Nglomeruli ** 2) ** 0.5,3)
   
    Media_accuracy= round(mean_accuracy_sum/num_lines,3)
    print(mean_accuracy_square,num_lines, Media_accuracy)
    STD_accuracy=(mean_accuracy_square / num_lines - Media_accuracy * Media_accuracy) ** 0.5
   

    Media_recall=round(mean_recall_sum/num_lines,3)
    STD_recall=round((mean_recall_square / num_lines - Media_recall ** 2) ** 0.5,3)
    
    Media_precision=round(mean_precision_sum/num_lines,3)
    STD_precision=round((mean_precision_square / num_lines - Media_precision ** 2) ** 0.5,3)
   
    Media_F1=round(mean_f1_sum/num_lines,3)
    STD_F1_score=round((mean_f1_square / num_lines - Media_F1 ** 2) ** 0.5,3)
   
    return Media_dice,STD_dice, Media_jaccard, STD_jaccard,Media_Nglomeruli,STD_Nglomeruli,Media_accuracy, \
        STD_accuracy, Media_recall,STD_recall,Media_precision, STD_precision ,Media_F1, STD_F1_score


Media_dice,STD_dice, Media_jaccard, STD_jaccard,Media_Nglomeruli,STD_Nglomeruli,Media_accuracy, \
    STD_accuracy, Media_recall,STD_recall,Media_precision, STD_precision ,Media_F1, STD_F1_score =mean(args.text_path)
print("Media:")
print("dice=",Media_dice) 
print("jaccard=",Media_jaccard)  
print("differenza glomeruli=",Media_Nglomeruli)  
print("accuracy=",Media_accuracy)    
print("recall=",Media_recall)    
print("precision=",Media_precision)    
print("F1 Score=",Media_F1)    
print("\n")
print("Deviazione Standard:")
print("dice=",STD_dice) 
print("jaccard=",STD_jaccard)  
print("differenza glomeruli=",STD_Nglomeruli)  
print("accuracy=",STD_accuracy)    
print("recall=",STD_recall)    
print("precision=",STD_precision)    
print("F1 Score=",STD_F1_score)    
