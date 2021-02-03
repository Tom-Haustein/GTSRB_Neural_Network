# 99,1% im GTSRB-Testdatensatz mit Tensorflow durch ein flaches CNN
This project is aviable on english too. Just follow this link to get to the englisch project.

Dieses Projekt beschäftigt sich mit der Entwicklung eines flachen CNN zur Erkennung von Verkehrsschildern. Das Projekt in Schriftform findet ihr als PDF unter Project.pdf.
Alles Programmcodes usw. findet ihr im weiteren hier. Das trainierte Model mit Aufbau und Gewichtungen findet ihr unter models / model_99,105%.hdf5.

# Umsetzung
Um das Ziel einer Verkehrsschilderkennung durch ein neuronales Netzwerk umzusetzen, habe ich in meinem Projekt die Bibliothek Tensorflow und Keras sowie einige weitere in Python genutzt. Wie Tensorflow genutzt und installiert wird, findet ihr hier.
Als Datensatz habe ich den GTSRB-Datensatz des Instituts für Neuroinformatik der TU Bochum genutzt. Dieser enthält rund 39000 Bilder im Testdatensatz. Eine ausführliche Analyse des Trainiungs- und Testdatensatzes findet ihr in der schriftlichen PDF-Datei, sowie alle Versuche zum Aufbau des CNN.

# Vorbereitung
Nach dem Download und entzippen des Datensatz (offizieller Downloadlink findet ihr hier) muss der Datensatz noch bearbeitet werden, damit dieser vom Trainingsprogramm fehlerfrei erkannt wird. Dafür habe ich folgendes Programm geschrieben, welches störende Datein löscht und die Ordner richtig umbenennt:
![Bild1.jpg](
# Training
Damit Bilder von einem neuronalen Netz gelernt werden können, müssen diese noch bearbeitet werden. Dazu müssen sie alle in die gleiche Bildgröße gebracht und anschließend als Tensor in den RAM geladen geladen werden. Die richtigen Labels zu den Trainingbilder werden aus den Ordnernamen entnommen und ebenfalls als Tensor geladen. Dies habe ich so umgesetzt:

Dieser Vorgang muss ebenfalls mit dem Testdatensatz durchgeführt werden. Dabei werden die richtigen Labels aus einer csv-Datei geladen. Der Testdatensatz ist nur dafür da, um zu beurteilen, wie genau das trainierte neuronale Netz tatsächlich ist. Im Vergleich zu vielen anderen ähnlichen Projekten mit dem GTSRB-Datensatz nutze ich den Testdatensatz und keinen Validiation Datensatz, in dem leicht höhere Ergebnisse erzielt werden können. Das CNN darf mit dem Testdatensatz auf keinen Falls trainiert werden, da dies das Ergebnis erheblich verfälschen würde. Das Laden des Testdatensatzes habe ich wie folgt gestaltet:

Da alle Trainings- und Testbilder nun geladen sind, geht es nun daran, unser neuronales Netz zu basteln und dieses dann zu trainieren. Den genutzten Aufbau und die Hyperparamteter, wie Optimizer usw. habe ich durch viele Versuchsreihen ermittelt. Das heißt, dass dieser Aufbau zwar gut funktioniert, es aber auch deutlich bessere Hyperparameterwahlen geben könnte. Leider wären unzählige Versuche nötig, um die optimale Kombination aus diesen zu finden. Wie ich zu einigen Teilen des Netzaufbaus durch Versuchsreihen gelangt bin, könnt ihr in der PDF nachlesen. Insagesamt wurden in dem ganzen Projekt rund 45000 Messdaten genommen. Dazu habe ich die Messwerte aus jeder Epoche in eine csv-Datei geschrieben und diese Daten anschließend ausgewertet. Ein solches Programm findet ihr unter Trainingsprogramm_mit_messdaten.py. Zurück zum eigentlichen Trainingsprogramm. Den Rest des Trainingsprgrammes könnt ihr hier sehen:  

Wie ihr vielleicht schon sehen konntet, wird dabei immer eine Epoche trainiert, dann im Testdatensatz getestet und wieder trainiert. Somit hat man alle Entwicklungen des Netzes im Blick. Falls eine Genauigkeit im Testdatensatz von über 99% erreicht wird, speichert das Programm das gesamte trainierte Netz als hdf5-Datei und trainiert anschließend weiter. Das Training sieht wie folgt aus:


Mit diesem Trainingprogramm erreichte im bereits ein Netz mit einer Genauigkeit von 98,8 % im Testdatensatz. Nun wollte ich explizit dieses trainierte Netz weiter tranieren, damit es sich noch weiter verbessert. Dafür habe ich ein zweites Programm zum Finetuning geschrieben.

# Finetuning
Meine Idee dazu war, dieses gespeicherte Model zu laden und zu trainieren bis es eine bestimmte Genauigkeit erreichte. Wenn dieses in einer bestimmten Epochenzahl diese nicht erreichte, wurde das neu trainierte Netz verworfen und das alte wieder geladen und neu trainiert. Dazu habe ich ein paar Hyperparamter verändert, zum Beispiel habe ich nun den Optimizer Adamax genutzt, da dieser für filigraneres Training besser geignet war und habe auch die Batchsize auf 64 erhöht. Den vollständigen Programmcode dazu könnt ihr hier sehen:

Durch immer leicht zufällige Ergebnisse ist diese Art des Trainings auch  gut umsetzbar. So kam ich stufenweise auf immer höhere Ergebnisse. Erst erreichte ich 99 % Genauigkeit, dann 99,04 % und schlussendlich eine Genauigkeit von 99,105 % (damit ist es rund 0,3% besser in der Verkehrsschildklassifikation als ein Mensch und ein internationales Top-Ergebnis). Dieses Trainingergebnis sah dann wie folgt aus:

# Test auf eigene Bilder
Nun hatte ich ein fertiges neuronalen Netz mit einer hohen Genauigkeit, da wollte ich auch testen, wie gut es tatsächlich ist. Dafür habe ich zehn Bilder von eigenen Verkehrsschildern aufgenommen. Aber gewöhnliche Verkehrsschilder wären ja viel zu langweilig. Deshalb habe ich besondere Verkehrsschilder fotografiert, die besonders schwer zu erkennen sind und stark von von den Trainingsbildern abweichen. Diese sahen dann beispielsweise so aus:

Im Ordner "eigene Bilder" findet ihr diese 10 Testbilder. Um diese zu testen, musste ich aber noch ein neues, kleines Programm schreiben, welches mir die Bilder klassifiziert. Dieses sieht dann wie folgt aus:

Dabei konnte mein trainiertes neuronales Netz alle dieser 10 Testbilder fehlerfrei richtig klassifizieren und konnte somit auch außerhalb des Testdatensatzes zeigen, dass Verkehrsschilder richtig klassifziert. 

# Live Verarbeitung
Damit ein neuronales Netz auch im Straßenverkehr eingesetzt werden könnte, muss es dauerhaft Verkehrsschilder klassifizieren. Aus dieser Intention heraus habe ich noch ein Programm geschrieben, mit welchem man das neuronale Netz in einer Live-Performance testen kann. Dabei liest es die Webcam des Computers aus und schickt diese Bilder in das neuronale Netz. Das Programm dazu sieht so aus:

Dabei brauchte das neuronale Netz eine durchschnittliche Klassifikationszeit von 14,9 ms vom Eingang des Bildes von der Webcam bis zur Klassifikation. Um die Ergebnisse optisch etwas ansprechend zu machen, habe ich es so geschrieben, dass man gleichzeitig 3 Bildfenster sehen kann. Auf der linken Seite sieht man die Bilder, welche die Webcam liefert. Auf der oberen rechten Seite sieht man, wie das erkannte Verkehrsschild aussieht (sodass man vergleichen kann) und direkt darunter werden Verkehrsschildname und Wahrscheinlichkeit angezeigt:

# Schluss
Ich hoffe ich konnte mit meinem Projekt vielen den Einstieg in die Welt der neuronalen Netze etwas erleichtern und ich hoffe ihr konntet etwas aus meinem Projekt mitnehmen. Bei Fragen und Anregungen könnt ihr mir auch schreiben.
