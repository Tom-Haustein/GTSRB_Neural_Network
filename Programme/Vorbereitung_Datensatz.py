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


