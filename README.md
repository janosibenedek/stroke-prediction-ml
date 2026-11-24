# stroke-prediction-ml

Stroke előrejelzése egészségügyi adatokból (bináris osztályozás): egy egyszerű
Baseline (logisztikus regresszió) és egy haladó modell (Random Forest)
statisztikai összehasonlítása. *MI adatelemzés (MSc) – féléves projekt.*

## Adat
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle):
5 110 sor, 10 jellemző. Töltsd le, és a `healthcare-dataset-stroke-data.csv`
fájlt tedd a `data/` mappába (az adat nem része a repónak).

## Futtatás
```bash
conda create -n stroke-ml python=3.12
conda activate stroke-ml
pip install -r requirements.txt
```
Utána futtasd le a `projekt.ipynb` notebookot (*Run All*).