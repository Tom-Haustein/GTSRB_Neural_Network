#importieren aller notwenigen Bibliotheken
import tensorflow.compat.v1 as tf
#Die Hauptbibliothek Tensorflow wird geladen
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import csv

Epochen=60
Trainingsbilder = []
Trainingslabels = []
print("Trainingsdaten werden geladen")
for i in range(0,43):
    n = str(i)
    Pfad = "GTSRB_Final_Training_Images/GTSRB/Final_Training/images/" + n
    label=i
    for Datei in os.listdir(Pfad):
     img = os.path.join(Pfad,Datei)
     #Bilder werden auf die Größe 32*32 Pixel mit RGB skaliert, damit diese eine einheitliche Größe haben
     img = image.load_img(img,target_size=(32,32))
     img = image.img_to_array(img,  dtype=np.float32)
     img=img.reshape(1,32,32,3)
     Trainingsbilder.append(img)
     Trainingslabels.append(label)
     #Doppeltes Hinzufügen der Trainingsbilder aus Bildklassen mit wenig Trainingsbildern
     if i==0 or i==6 or i==18 or i==16 or i==19 or i==20 or i==21 or i==24 or i==27 or i==29 or i==32 or i==37:
         Trainingsbilder.append(img)
         Trainingslabels.append(label)

#Umformung der Liste mit den Trainingsbildern in einen Tensor
Trainingslabels = np.asarray(Trainingslabels)
Trainingsbilder = np.asarray([Trainingsbilder])
Trainingsbilder = Trainingsbilder.reshape(-1, 32, 32, 3)
#Umwandlung der Farbwerte in Gleitkommazahlen zwischen 0 und 1
Trainingsbilder = Trainingsbilder/255
Trainingsbilder = np.asarray(Trainingsbilder, dtype = "float32")
Trainingslabels = np.asarray(Trainingslabels, dtype= "float32")


Testbilder = []
Testlabels = []
print()
print("Testdaten werden geladen")

#Laden der Testbilder  als deren Bildtensoren in eine Liste
Testpfad="GTSRB_Final_Test_Images/GTSRB/Final_Test/images/"
for Datei in os.listdir(Testpfad):
     img = os.path.join(Testpfad,Datei)
     #Umformung der Testbilder in die Größe 32*32 Pixel
     img = image.load_img(img,target_size=(32,32))
     img = image.img_to_array(img,  dtype=np.float32)
     img = img.reshape(1,32,32, 3)
     Testbilder.append(img)

#Auslesen der richtigen Bildklassen der Testbilder aus einer CSV-Datei

with open('Testdaten.csv') as csvdatei:
    csv_datei = csv.reader(csvdatei)
    for Reihe in csv_datei:
        Testlabels.append(Reihe[6])

#Umformung der Liste mit den Testbildern in einen Tensor
Testlabels.pop(0)    
Testlabels = np.asarray(Testlabels)
Testbilder = np.asarray([Testbilder])
Testbilder = Testbilder.reshape(-1, 32, 32, 3)
#Umwandlung der Farbwerte in Gleitkommazahlen zwischen 0 und 1
Testbilder = Testbilder/255
Testbilder = np.asarray(Testbilder, dtype = "float32")
Testlabels = np.asarray(Testlabels, dtype= "float32")

#Zusammenstellen des Neuronalen Netzes
#zuerst Zusammenstellen der Filter mit Batchnormalisierung (3 Convolutional Filter, 2 Pooling Filter)

model = Sequential(name='CNN')
model.add(Conv2D(32, (3, 3), activation='selu', padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#Umformung des Veränderten Tensors in einen langen Vektor
model.add(Flatten())

#Aufstellen der 3 Neuronenschichten mit 750, 256 und 43 Neuronen, Festlegen der Dropoutraten
#Neuronenzahl der 1. Schicht
model.add(Dense(750))
#Aktivierungsfunktion relu
model.add(Activation('relu'))
#Dropout festlegen
model.add(Dropout(0.4))
#Batchnormalisierung
model.add(BatchNormalization())
#weitere Schichten
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(43))
#Softmax zur Umwandlung in Klassenwahrscheinlichkeiten
model.add(Activation('softmax'))

#festlegen von Verlustfunktion, Optimizer und metrics
model.compile(loss='sparse_categorical_crossentropy',
 optimizer='Adam',
 metrics=['accuracy'])

#Befehl zum Trainieren des Netzes über 60 Epochen mit shuffle, Batchsize 32
#Trainiert wird mit den Trainingsbildern, nach jeder Trainingsepoche wird auf die Genauigkeit im Testdatensatz getestet
for i in range(Epochen):
    model.fit(Trainingsbilder, Trainingslabels, epochs=1, shuffle=True, batch_size=32)
    #aus den Ergebnissen wird eine Genauigkeit im Testdatensatz errechnet, sowie ein durchschnittlicher Verlust
    score=model.evaluate(Testbilder, Testlabels)
    
    print('Epoche',i+1)
    print('Test Verlust:', score[0])
    print('Test Genauigkeit:', score[1])
    #speichern des trainierten Models im hdf5-Format, falls es über 99% Genauigkeit im testdatensatz hat
    if score[1]>0.99:
        model.save('model_'+str(score[1])+'.hdf5')
        print("gespeichert")
        
