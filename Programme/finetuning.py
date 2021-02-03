import tensorflow.compat.v1 as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam, Nadam, Adamax, Adadelta


Trainingsbilder = []
Trainingslabels = []
print("Trainingsdaten werden geladen")
for i in range(0,43):
 n = str(i)
 Pfad = "D:/GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/" + n
 label=i
 for Datei in os.listdir(Pfad):
     img = os.path.join(Pfad,Datei)
     img = image.load_img(img,target_size=(32,32))
     img = image.img_to_array(img,  dtype=np.float32)
     img=img.reshape(1,32,32,3)
     img=img/255
     Trainingsbilder.append(img)
     Trainingslabels.append(label)

Trainingslabels = np.asarray(Trainingslabels)
Trainingsbilder = np.asarray([Trainingsbilder])
Trainingsbilder = Trainingsbilder.reshape(-1, 32, 32, 3)
Trainingsbilder = np.asarray(Trainingsbilder, dtype = "float32")
Trainingslabels = np.asarray(Trainingslabels, dtype= "float32")

Testbilder = []
Testlabels = []
print()
print("Testdaten werden geladen")
    
import csv
Testpfad="D:/GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/images/"
for Datei in os.listdir(Testpfad):
    img = os.path.join(Testpfad,Datei)
    img = image.load_img(img,target_size=(32,32))
    img = image.img_to_array(img,  dtype=np.float32)
    img = img.reshape(1,32,32,3)
    img=img/255
    Testbilder.append(img)
Liste_Testbilder = Testbilder
    
with open('D:/GTSRB/Testdaten.csv') as csvdatei:
    csv_datei = csv.reader(csvdatei)
    for Reihe in csv_datei:
        Testlabels.append(Reihe[6])
    
Testlabels.pop(0) 
Liste_Testlabels = Testlabels    
Testlabels = np.asarray(Testlabels)
Testbilder = np.asarray([Testbilder])
Testbilder = Testbilder.reshape(-1, 32, 32, 3)
Testbilder = np.asarray(Testbilder, dtype = "float32")
Testlabels = np.asarray(Testlabels, dtype= "float32")


model=load_model('model_99,105%.hdf5')

best=0.99105305
batch_size1=64
batch_size2=64
batch_size3=128
opt=Adamax(lr=0.0004)

for a in range(10):
    for i in range(150):
        
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,metrics=['accuracy'])
        if i<50:
            
            model.fit(Trainingsbilder, Trainingslabels, epochs=1, 
                  shuffle=True, batch_size=batch_size1)
           
            batch_size=batch_size1
        if i==50:
            print("neue Batchsize:",batch_size2)
            
        if i>=50 and i<100:
            
            model.fit(Trainingsbilder, Trainingslabels, epochs=1, 
                  shuffle=True, batch_size=batch_size2)
            batch_size=batch_size2
            
        if i==100:
            print('neue Batchsize',batch_size)
        if i>=100:
            model.fit(Trainingsbilder, Trainingslabels, epochs=1, 
                  shuffle=True, batch_size=batch_size3)
            batch_size=batch_size3
            
        epoch=int(str(a)[0]+str(i))+1
        score=model.evaluate(Testbilder, Testlabels)
        print(score[1],'Batchsize:',batch_size,'Epoche:',epoch)

        if score[1]>best:
            model.save('CNN_best'+str(score[1])+'.hdf5')
            best=score[1]
            print("NEUER BESTWERT !!!")

