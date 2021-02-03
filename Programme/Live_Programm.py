def bilderladen():
     Pfad = "D:/GTSRB/Beispielbilder"
     for Datei in os.listdir(Pfad):
         img = os.path.join(Pfad,Datei)
         img=cv2.imread(img)
         img=cv2.resize(img,(450,450))
         img=img/255
         Bilder.append(img)

Bilder=[]

import tensorflow.compat.v1 as tf
import time
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


vk = ['20 km/h Tempolimit','30 km/h Tempolimit','50 km/h Tempolimit','60 km/h Tempolimit','70 km/h Tempolimit','80 km/h Tempolimit','Ende des 80 km/h Temolimits','100 km/h Tempolimit','120 km/h Tempolimit','allgemeines Ueberholverbot','Ueberholverbot für LKW','Einzelvorfahrt','Vorfahrt','Vorfahrt gewaeren','Stopschild','Sperrscheibe','keine Durchfahrt für LKW','keine Durchfahrt','Achtung','Achtung scharfe Linkskurve','Achtung scharfe Rechtskurve', 'Achtung scharfe Doppelkurve','Achtung Bodenwellen','Achtung Rutschgefahr','Achtung Fahrbahnverengung rechts','Achtung Bauarbeiten','Achtung Ampel','Achtung Fußgaenger','Achtung Kinder','Achtung Fahrraeder','Achtung Schneefall','Achtung Wildwechsel','Aufhebung der Geschwindigkeitsbegrenzung','Zwangspfeil rechts','Zwangspfeil links','Zwangspfeil gerade','Zwangspfeil gerade oder rechts','Zwangspfeil gerade oder links','Hier rechts vorbei fahren','Hier links vorbei fahren','Kreisverkehr','Aufhebung Ueberholverbot','Aufhebung Ueberholverbot für LKW']
bilderladen()

c=111
d=0
model=load_model('CNN_best0.99105304.hdf5')
vid = cv2.VideoCapture(0) 
while(True): 
    ret, frame = vid.read()  
    cv2.imshow('Bild', frame)
    frame = cv2.resize(frame,(32,32))
    frame = frame/255
    frame = frame.reshape(1,32,32,3)
    classes = model.predict(frame)
    a = np.argmax(classes[0])
    b = np.amax(classes[0])
    if a!=c or round(b,2)!=round(d,2):
        c=a
        bild_text=np.zeros((150,650,3),np.uint8)
        cv2.putText(bild_text,str(vk[a]),(60,30),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
        cv2.putText(bild_text,str((round(b*100,2)))+'%',(60,90),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
        cv2.imshow('Verkehrsschild',bild_text)
        cv2.imshow('erkanntes Verkehrsschild',Bilder[a])
        d=b
    if cv2.waitKey(20) & 0xFF == ord('q'): 
        break
vid.release() 

cv2.destroyAllWindows() 
