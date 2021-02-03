#Importieren aller Bibliotheken
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

#Laden des tranierten neuronalen netzes mit Gewichten 
model=load_model('CNN_best0.9904196.hdf5')

#Auswählen eines Einzelbildes, das klassifiziert werden soll
#für Testen eines anderen Testbildes in der nachfolgenden Zeile den Namen des gwünschten Bildes eingeben
#alle Bilder im testordner sind selbst aufgenommen und zugeschnitten
bild="Bild10.jpg"
test_img = "Testbilder/"+bild

#Testbild wird das richtige Format von 32*32 Pixel umgeformt
test_img = os.path.join(test_img)
test_img = image.load_img(test_img,target_size=(32,32))
test_img = image.img_to_array(test_img,  dtype=np.float32)
test_img = test_img/255
plt.imshow(test_img)
test_img = test_img.reshape(1,32,32,3)

#skaliertes Bild wird gezeigt
plt.show()

#Bild wird durch das neuronale Netz geschickt und somit klassifiziert
classes = model.predict(test_img)
#Verteilung der Klassenwahrscheinlichkeit wird als diagramm angezeigt
plt.bar(range(43), classes[0])
plt.show()

#Ergebnis mit der höchsten wahrscheinlichkeit wird ausgegeben
a = np.argmax(classes[0])
b = np.amax(classes[0])
print("Vorhersage: Bildklasse",a+1,"mit einer WK von",b*100,"%")
#(a+1 da Bildklasse 1 im Programm Bildklasse 0 heißt)
verkehrsschild = ['20 km/h Tempolimit','30 km/h Tempolimit','50 km/h Tempolimit','60 km/h Tempolimit','70 km/h Tempolimit','80 km/h Tempolimit','Ende des 80 km/h Temolimits','100 km/h Tempolimit','120 km/h Tempolimit','allgemeines Ueberholverbot','Ueberholverbot für LKW','Einzelvorfahrt','Vorfahrt','Vorfahrt gewaeren','Stopschild','Sperrscheibe','keine Durchfahrt für LKW','keine Durchfahrt','Achtung','Achtung scharfe Linkskurve','Achtung scharfe Rechtskurve', 'Achtung scharfe Doppelkurve','Achtung Bodenwellen','Achtung Rutschgefahr','Achtung Fahrbahnverengung rechts','Achtung Bauarbeiten','Achtung Ampel','Achtung Fußgaenger','Achtung Kinder','Achtung Fahrraeder','Achtung Schneefall','Achtung Wildwechsel','Aufhebung der Geschwindigkeitsbegrenzung','Zwangspfeil rechts','Zwangspfeil links','Zwangspfeil gerade','Zwangspfeil gerade oder rechts','Zwangspfeil gerade oder links','Hier rechts vorbei fahren','Hier links vorbei fahren','Kreisverkehr','Aufhebung Ueberholverbot','Aufhebung Ueberholverbot für LKW']
print('Bildklasse',a+1,': ',verkehrsschild[a])
#Ausgabe Verkehrsschildname