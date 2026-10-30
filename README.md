# Predikcija cene polovnih automobila

Regresioni projekat koji procenjuje cenu polovnog automobila u USD na osnovu marke, modela, godišta, kilometraže, stanja i drugih karakteristika. Podaci su dati u `data/cars.csv`.

## Struktura

```text
car-price-prediction/
├── data/cars.csv
├── notebooks/01_eda_and_modeling.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   └── predict.py
├── models/car_price_model.joblib
├── reports/
├── requirements.txt
└── README.md
```

## Pokretanje

Potrebni su Python 3.10 ili noviji. Iz osnovnog direktorijuma projekta pokreni:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/data_cleaning.py
python src/model_training.py
python src/model_evaluation.py
```

Skripta za treniranje sama obavlja čišćenje i inženjering karakteristika, pravi isti reproducibilan 80/20 train/test split za sve modele i čuva najbolji pipeline. `data_cleaning.py` je izdvojen i može se pokrenuti samostalno; generiše `data/cars_clean.csv`.

Za pregled analize otvori `notebooks/01_eda_and_modeling.ipynb` u Jupyter Lab-u ili Notebook-u. Za poređenje pokreni `python src/model_comparison.py` (isto treniranje, pet modela). Rezultati se upisuju u `reports/model_comparison.csv`, `reports/sample_predictions.csv` i `reports/training_summary.json`.

## Javni GitHub repozitorijum

Projekat je objavljen javno: https://github.com/nikolicure35-cloud/car-price-prediction

Za kopiju projekta pokreni `git clone https://github.com/nikolicure35-cloud/car-price-prediction.git`.

## Predikcija za novi automobil

Kad je model istreniran:

```bash
python src/predict.py
```

Ili pozovi funkciju iz Python-a:

```python
from src.predict import predict_price

car = {
    "make": "volkswagen", "model": "golf", "year": 2014,
    "condition": "with mileage", "mileage_km": 180000,
    "fuel_type": "diesel", "volume_cm3": 1600, "color": "black",
    "transmission": "mechanics", "drive_unit": "front-wheel drive", "segment": "c"
}
print(predict_price(car))
```

Funkcija prihvata i izvorne nazive `mileage(kilometers)` i `volume(cm3)`. Ne unosi se `priceUSD` prilikom predikcije.

## Obrada podataka i odluke

- Cilj je `priceUSD`, preimenovan u `price_usd` nakon učitavanja.
- Izbačeni su samo redovi bez validne, pozitivne ciljne cene. Sumnjivo godište, kilometraža ili zapremina tretiraju se kao nedostajuće, a ne brišu ceo oglas.
- Nedostajuće numeričke vrednosti popunjavaju se medianom, kategorijske najčešćom vrednošću. Nepoznate kategorije pri predikciji se prihvataju bez greške.
- Izvedene karakteristike: starost automobila u odnosu na godinu treniranja, litri motora, prosečna kilometraža godišnje, indikator velike kilometraže i spoj marke/modela.
- Koriste se kategorijske kolone sa OneHotEncoder-om i numeričke sa imputacijom i skaliranjem. Imputacija i kodiranje se uče samo na trening delu kroz sklearn Pipeline.
- Ekstremno skupe i jeftine, ali pozitivne cene nisu automatski odbačene. One mogu biti stvarni oglasi; zato se MAE i primeri greške tumače uz raspon cena.

## Modeli i evaluacija

Upoređuju se Ridge Regression, Decision Tree, Random Forest, Extra Trees i HistGradientBoosting. Svi koriste isti nasumični hold-out skup (20%, seed 42). Glavna metrika izbora je MAE, uz RMSE, MSE i R². MAE se izražava u dolarima i predstavlja prosečnu apsolutnu grešku predikcije.

Na fiksnoj podeli 80/20 (seed 42), Extra Trees je dao najmanji MAE. Test skup ima 11.249 oglasa.

| Model | MAE ($) | RMSE ($) | R² |
|---|---:|---:|---:|
| Extra Trees | 1,115 | 2,727 | 0.887 |
| Random Forest | 1,150 | 2,807 | 0.880 |
| HistGradientBoosting | 1,264 | 2,856 | 0.876 |
| Decision Tree | 1,349 | 3,258 | 0.838 |
| Ridge Regression | 2,202 | 4,371 | 0.709 |

Izbor Extra Trees znači da je prosečna apsolutna greška na test skupu oko $1.115 po oglasu. Veći RMSE ukazuje da postoje i znatno veći pojedinačni promašaji, naročito kod skupljih vozila. Detaljne metrike i primeri su u `reports/`. Finalni model sa pretprocesiranjem čuva se u `models/car_price_model.joblib`.

## Ograničenja

Rezultat je procena zasnovana na ovom skupu oglasa, a ne zvanična tržišna procena. Nasumična podela meri performanse na sličnim podacima iz istog skupa; za procenu budućih tržišnih cena bolja bi bila vremenska validacija sa datumom oglasa. Model ne koristi lokaciju, servisnu istoriju, opremu, oštećenja niti stvarnu cenu transakcije. Preporuka je da se predikcija koristi kao orijentir i prikaže sa opsegom/objašnjenjem greške.
