# SpotifyModel
Machine Learning model that is trained upon data from Spotify

# Analiza e Karakteristikave Audio në Spotify (1980 - 2025)

Ky projekt analizon këngët më të njohura në Spotify nga viti 1980 deri në 2025 për të parë si kanë ndryshuar karakteristikat audio me kalimin e kohës. Qasja është hibride:

- Dataset Kaggle (1921-2020), i filtruar në periudhën 1980-2020.
- Spotify API për periudhën 2021-2025.

Qëllimi është të ndërtohet një analizë statistikore dhe një pipeline i machine learning për të kuptuar evolucionin e muzikës moderne.

## Pyetjet kryesore të kërkimit

1. Si kanë ndryshuar karakteristikat kryesore audio nga 1980 deri në 2025?
2. A ekziston tendencë drejt standardizimit të muzikës moderne?
3. Cilat janë marrëdhëniet ndërmjet karakteristikave audio?
4. Cilat karakteristika ndikojnë më shumë në popullaritet?
5. A mund të grupohen këngët në kategori sipas profilit audio?
6. A ka ndryshime të dukshme para dhe pas vitit 2010 (epoka e streaming)?

## Variablat që do të analizohen

- danceability
- energy
- loudness
- tempo
- valence
- acousticness
- instrumentalness
- speechiness
- liveness
- duration_ms
- popularity (target për modelin e parashikimit)

## Plani i analizës (sipas pyetjeve)

### 1) Trendet ndër dekada (1980-2025)

Metodologjia:

- Ndarja në dekada: 1980s, 1990s, 2000s, 2010s, 2020-2025.
- Llogaritje e statistikave për çdo variabël: mesatare, medianë, devijim standard.
- Vizualizime: line plots sipas vitit dhe boxplots sipas dekadës.
- Teste trendi: regresion linear i variablës ndaj vitit dhe, kur duhet, test jo-parametrik.

Output i pritshëm:

- Grafikë që tregojnë nëse p.sh. danceability dhe loudness rriten me kohën.
- Tabelë përmbledhëse me ndryshimin e secilës variabël ndër dekada.

### 2) Standardizimi i muzikës moderne

Metodologjia:

- Matje e variancës dhe IQR për çdo variabël në çdo dekadë.
- Krahasim i shpërndarjeve (density plots/violin plots) mes dekadave.
- Teste për barazi variancash (p.sh. Levene) për periudhat kryesore.

Interpretimi:

- Ulje e variancës me kohën sugjeron standardizim.
- Shpërndarje më e ngushtë në vitet e fundit tregon homogjenizim të profilit audio.

### 3) Korrelacionet ndërmjet karakteristikave audio

Metodologjia:

- Matrica e korrelacionit (Pearson dhe Spearman).
- Heatmap globale për të gjithë periudhën dhe heatmaps të ndara sipas dekadës.
- Analizë e ndryshimit të korrelacioneve me kohën.

Output i pritshëm:

- Lidhjet më të forta pozitive/negative (p.sh. energy me loudness).
- Evidencë nëse marrëdhëniet strukturore të muzikës ndryshojnë ndër dekada.

### 4) Modeli ML për parashikimin e popullaritetit

Target:

- popularity.

Features:

- Të gjitha karakteristikat audio + year (opsionale si feature).

Hapat:

- Train/validation/test split me logjikë kohore për të shmangur leakage.
- Baseline model: Linear Regression.
- Modele më të forta: Random Forest Regressor dhe XGBoost/Gradient Boosting.
- Metrika: MAE, RMSE, R2.
- Interpretueshmëria: feature importance + SHAP values (nëse përdoret model pemësh).

Output i pritshëm:

- Renditja e karakteristikave që ndikojnë më shumë në popullaritet.
- Krahasim i performancës së modeleve.

### 5) Clustering i këngëve sipas karakteristikave audio

Metodologjia:

- Standardizim i features.
- Algoritmi kryesor: KMeans.
- Vlerësim i numrit optimal të grupeve me elbow method dhe silhouette score.
- Alternativë: GMM ose DBSCAN për krahasim.
- Vizualizim në 2D me PCA/UMAP.

Analiza ndër dekada:

- Përqindja e çdo cluster-i në secilën dekadë.
- Evolucioni i profileve dominuese të këngëve me kalimin e kohës.

### 6) Krahasimi para dhe pas 2010 (epoka e streaming)

Ndarja e periudhave:

- Para streaming: 1980-2009.
- Pas streaming: 2010-2025.

Metodologjia:

- Krahasim mesataresh/medianash për secilën variabël.
- Teste domethënieje (t-test ose Mann-Whitney sipas shpërndarjes).
- Matje e madhësisë së efektit (Cohen's d).
- Krahasim i variancës për të parë ndryshimin në diversitet.

Output i pritshëm:

- Evidencë statistikore për dallimet strukturore të muzikës në epokën e streaming.

## Pipeline i implementimit

1. Data ingestion

- Ngarkimi i Kaggle dataset dhe filtrimi 1980-2020.
- Marrja e të dhënave 2021-2025 nga Spotify API.

2. Data cleaning dhe harmonization

- Unifikim i kolonave dhe tipeve.
- Heqje dublikatash.
- Menaxhim i vlerave mungese dhe outliers.

3. Exploratory Data Analysis

- Trende, shpërndarje, variancë, korrelacione.

4. Statistical testing

- Teste për trend, variancë dhe krahasim periudhash.

5. Modeling

- Regresion për popularity.
- Clustering për tipologjitë e këngëve.

6. Reporting

- Notebook + grafikë + përfundime kryesore.

## Deliverables

- Dataset final i unifikuar 1980-2025.
- Raport analitik me grafikë dhe teste statistikore.
- Model ML me metrika performance dhe interpretim feature importance.
- Analizë clustering me evolucionin e grupeve ndër dekada.
- Seksion përfundimtar: çfarë ka ndryshuar në muzikë dhe sa është standardizuar ajo.


## Ekzekutimi i Pipeline-it të Plotë

Instalo varësitë e projektit:
```powershell
pip install -r requirements.txt

Krijo skedarin .env në rrënjën e projektit me kredencialet e Spotify API:
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

1. Shkarko dhe përgatit të dhënat e Kaggle (1980-2020)

python -m src.data.ingest_kaggle `
  --input-csv data/interim/kaggle_1960_2020.csv `
  --output-csv data/interim/kaggle_prepared.csv


2. Merr të dhënat e Spotify API (2021-2025)

python -m src.data.ingest_spotify `
  --output-csv data/raw/spotify/tracks_2021_2025.csv `
  --api-start-year 2021 `
  --api-end-year 2025 `
  --top-n 50

`--fix-csv` mbush mungesat e "popularity" në një skedar CSV ekzistues


3. Bashko të dhënat e Kaggle dhe Spotify në një tabelë të unifikuar

python -m src.data.merge_datasets `
  --kaggle-csv data/interim/kaggle_prepared.csv `
  --spotify-csv data/raw/spotify/tracks_2021_2025.csv `
  --output-csv data/processed/Spotify_1980_2025_Final.csv


4. Pastroni tëdhënat: heq dublikatat, menaxho vlerat mungese, normalizimi

python src/data/finalize_cleaning.py `
  --input data/processed/Spotify_1980_2025_Final.csv `
  --output data/processed/Spotify_1980_2025_Final_clean.csv


5. Përgatit features për modelimin: ndarje e features dhe target (popularity)

python -m src.features.preprocess

Prodhon: data/processed/model_input.csv


6. Trajno modelet e machine learning për parashikimin e popullaritetit

python -m src.models.train_popularity_model


7. Grupo këngët sipas karakteristikave audio me KMeans clustering

python -m src.models.clustering_analysis


8. Krijo analiza statistikore (trendet, korrelacionet, krahasimi para/pas 2010)

python -m src.analysis.trend_analysis
python -m src.analysis.correlation_analysis
python -m src.analysis.standardization_analysis
python -m src.analysis.pre_post_2010_analysis
python -m src.analysis.create_spotify_visualizations


9. Dashboard-it Interaktiv

streamlit run dashboards\dashboard.py

Start-Process "c:\Users\Asus\OneDrive\Desktop\SpotifyModel\reports\figures\figure_4_interactive_scatter.html"


Rezultatet
Pas përfundimit të pipeline-it, të gjithë rezultatet do të jenë në:

- reports — grafikë, metrikat e modeleve, raporte analitike
- models/ — modelet e trajnuara (joblib files)
- processed — të dhënat e përpunuara dhe të gatshme për analizë