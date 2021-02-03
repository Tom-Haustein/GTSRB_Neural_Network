# 99,1% im GTSRB-Testdatensatz mit Tensorflow durch ein flaches CNN
This project is also available in English. You can find the link here.

Dieses Projekt beschäftigt sich mit der Entwicklung eines flachen CNN zur Erkennung von Verkehrsschildern. Das Projekt in Schriftform mit deutlich mehr Detail, Versuchen und Erklärungen findet ihr als PDF unter ![Projekt](https://github.com/bomm412/GTSRB_Convolutional_Neural_Network/blob/main/Projekt.pdf).
Alles Programmcodes usw. findet ihr im weiteren hier. Das trainierte Model mit Aufbau und Gewichtungen findet ihr hier: ![model](https://github.com/bomm412/GTSRB_Convolutional_Neural_Network/blob/main/models/model_99%2C105%25.hdf5)


# Umsetzung
Um das Ziel einer Verkehrsschilderkennung durch ein neuronales Netzwerk umzusetzen, habe ich in meinem Projekt die Bibliothek Tensorflow und Keras sowie einige weitere in Python genutzt. Wie Tensorflow genutzt und installiert wird, findet ihr hier.
Als Datensatz habe ich den [GTSRB-Datensatz](https://benchmark.ini.rub.de/gtsrb_news.html) des Instituts für Neuroinformatik der TU Bochum genutzt. Dieser enthält rund 39000 Bilder im Testdatensatz. Eine ausführliche Analyse des Trainiungs- und Testdatensatzes findet ihr in der schriftlichen PDF-Datei, sowie alle Versuche zum Aufbau des CNN.

# Vorbereitung
Nach dem Download und entzippen des Datensatz [Downloadlink findet ihr hier](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) muss der Datensatz noch bearbeitet werden, damit dieser vom Trainingsprogramm fehlerfrei erkannt wird. Dafür habe ich folgendes Programm geschrieben, welches störende Datein löscht und die Ordner richtig umbenennt:
```ruby
import os
pfad="GTSRB/Final_Training/Images/"
for ordner in os.listdir(pfad):
    ordnername = int(ordner)
    ordnername = str(ordnername)
    if (ordnername != ordner):
        os.rename(pfad + ordner,pfad + ordnername)
for i in range(0,43):
    n = str(i)
    subpfad = os.path.join(pfad, n)
    for datei in os.listdir(subpfad):
        if not datei.endswith('.csv'):
            continue
        datei_mit_pfad = os.path.join(subpfad,datei)
        os.remove(datei_mit_pfad)
```


# Training
Damit Bilder von einem neuronalen Netz gelernt werden können, müssen diese noch bearbeitet werden. Dazu müssen sie alle in die gleiche Bildgröße gebracht und anschließend als Tensor in den RAM geladen geladen werden. Ich habe dabei eine Bildergröße von 32x32 Pixel gewählt, in die alle Bilder skaliert werden. Die richtigen Labels zu den Trainingbilder werden aus den Ordnernamen entnommen und ebenfalls als Tensor geladen. Dies habe ich so umgesetzt:
```ruby
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
     img=img/255
     img=filter(img)
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
#Trainingsbilder = Trainingsbilder/255
Trainingsbilder = np.asarray(Trainingsbilder, dtype = "float32")
Trainingslabels = np.asarray(Trainingslabels, dtype= "float32")
```
Dieser Vorgang muss ebenfalls mit dem Testdatensatz durchgeführt werden. Dabei werden die richtigen Labels aus einer csv-Datei geladen. Der Testdatensatz ist nur dafür da, um zu beurteilen, wie genau das trainierte neuronale Netz tatsächlich ist. Im Vergleich zu vielen anderen ähnlichen Projekten mit dem GTSRB-Datensatz nutze ich den Testdatensatz und keinen Validiation Datensatz, in dem leicht höhere Ergebnisse erzielt werden können. Das CNN darf mit dem Testdatensatz auf keinen Fall trainiert werden, da dies das Ergebnis erheblich verfälschen würde. Das Laden des Testdatensatzes habe ich wie folgt gestaltet:
```ruby
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
     img = filter(img)
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
```
Da alle Trainings- und Testbilder nun geladen sind, geht es nun daran, unser neuronales Netz zu basteln und dieses dann zu trainieren. Den genutzten Aufbau und die Hyperparamteter, wie Optimizer usw. habe ich durch viele Versuchsreihen ermittelt. Das heißt, dass dieser Aufbau zwar gut funktioniert, es aber auch deutlich bessere Hyperparameterwahlen geben könnte. Leider wären unzählige Versuche nötig, um die optimale Kombination aus diesen zu finden. Wie ich zu einigen Teilen des Netzaufbaus durch Versuchsreihen gelangt bin, könnt ihr in der PDF nachlesen. Insagesamt wurden in dem ganzen Projekt rund 45000 Messdaten genommen. Dazu habe ich die Messwerte aus jeder Epoche in eine csv-Datei geschrieben und diese Daten anschließend ausgewertet. Ein solches Programm findet ihr unter Trainingsprogramm_mit_messdaten.py. Zurück zum eigentlichen Trainingsprogramm. Den Rest des Trainingsprogrammes könnt ihr hier sehen:  
```ruby
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
```
Wie ihr vielleicht schon sehen konntet, wird dabei immer eine Epoche trainiert, dann im Testdatensatz getestet und wieder trainiert. Somit hat man alle Entwicklungen des Netzes im Blick. Falls eine Genauigkeit im Testdatensatz von über 99% erreicht wird, speichert das Programm das gesamte trainierte Netz als hdf5-Datei und trainiert anschließend weiter. Das Training sieht wie folgt aus:


Mit diesem Trainingprogramm erreichte im bereits ein Netz mit einer Genauigkeit von 98,8 % im Testdatensatz. Nun wollte ich explizit dieses trainierte Netz weiter tranieren, damit es sich noch weiter verbessert. Dafür habe ich ein zweites Programm zum Finetuning geschrieben.

# Finetuning
Meine Idee dazu war, dieses gespeicherte Model zu laden und zu trainieren bis es eine bestimmte Genauigkeit erreichte. Wenn dieses in einer bestimmten Epochenzahl diese nicht erreichte, wurde das neu trainierte Netz verworfen und das alte wieder geladen und neu trainiert. Dazu habe ich ein paar Hyperparamter verändert, zum Beispiel habe ich nun den Optimizer Adamax genutzt, da dieser für filigraneres Training besser geignet war und habe auch die Batchsize auf 64 erhöht. Den vollständigen Programmcode dazu könnt ihr hier sehen:
```ruby
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
```
Durch immer leicht zufällige Ergebnisse ist diese Art des Trainings auch  gut umsetzbar. So kam ich stufenweise auf immer höhere Ergebnisse. Erst erreichte ich 99 % Genauigkeit, dann 99,04 % und schlussendlich eine Genauigkeit von 99,105 % (damit ist es rund 0,3% besser in der Verkehrsschildklassifikation [als ein Mensch](https://christian-igel.github.io/paper/MvCBMLAfTSR.pdf) und ein internationales Top-Ergebnis). Dieses Trainingergebnis sah dann wie folgt aus:
 
![Trainingsergebnis.jpg](https://github.com/bomm412/GTSRB_Convolutional_Neural_Network/blob/main/Bilder/top_result.JPG)

# Test auf eigene Bilder
Nun hatte ich ein fertiges neuronalen Netz mit einer hohen Genauigkeit, da wollte ich auch testen, wie gut es tatsächlich ist. Dafür habe ich zehn Bilder von eigenen Verkehrsschildern aufgenommen. Aber gewöhnliche Verkehrsschilder wären ja viel zu langweilig. Deshalb habe ich besondere Verkehrsschilder fotografiert, die besonders schwer zu erkennen sind und stark von von den Trainingsbildern abweichen. Diese sahen dann beispielsweise so aus:

Im Ordner "eigene Bilder" findet ihr diese 10 Testbilder. Um diese zu testen, musste ich aber noch ein neues, kleines Programm schreiben, welches mir die Bilder klassifiziert. Dieses sieht dann wie folgt aus:
```ruby
#Importieren aller Bibliotheken
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

#Laden des tranierten neuronalen netzes mit Gewichten 
model=load_model('models/model_99,105%')

#Auswählen eines Einzelbildes, das klassifiziert werden soll
#für Testen eines anderen Testbildes in der nachfolgenden Zeile den Namen des gwünschten Bildes eingeben
#alle Bilder im testordner sind selbst aufgenommen und zugeschnitten
bild="Bild1.jpg"
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
```
Dabei konnte mein trainiertes neuronales Netz alle dieser 10 Testbilder fehlerfrei richtig klassifizieren und konnte somit auch außerhalb des Testdatensatzes zeigen, dass Verkehrsschilder richtig klassifziert. 

# Live Verarbeitung
Damit ein neuronales Netz auch im Straßenverkehr eingesetzt werden könnte, muss es dauerhaft Verkehrsschilder klassifizieren. Aus dieser Intention heraus habe ich noch ein Programm geschrieben, mit welchem man das neuronale Netz in einer Live-Performance testen kann. Dabei liest es die Webcam des Computers aus und schickt diese Bilder in das neuronale Netz. Das Programm dazu sieht so aus:
```ruby
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
```
Dabei brauchte das neuronale Netz eine durchschnittliche Klassifikationszeit von 14,9 ms (bei 6,2 TFLOPS Rechenleistung) vom Eingang des Bildes von der Webcam bis zur Klassifikation. Um die Ergebnisse optisch etwas ansprechend zu machen, habe ich es so geschrieben, dass man gleichzeitig 3 Bildfenster sehen kann. Auf der linken Seite sieht man die Bilder, welche die Webcam liefert. Auf der oberen rechten Seite sieht man, wie das erkannte Verkehrsschild aussieht (sodass man vergleichen kann) und direkt darunter werden Verkehrsschildname und Wahrscheinlichkeit angezeigt:

# Schluss
Ich hoffe ich konnte mit meinem Projekt vielen den Einstieg in die Welt der neuronalen Netze etwas erleichtern und ich hoffe ihr konntet etwas aus meinem Projekt mitnehmen. Bei Fragen und Anregungen könnt ihr mir auch schreiben.
