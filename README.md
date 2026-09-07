# StayType AI — NYC Airbnb Room Type Predictor

A production machine learning web application that classifies NYC Airbnb listings into **Entire home/apt**, **Private room**, or **Shared room** using real listing data — deployed as a live, interactive tool.

**🌐 Live Application:** [https://nyc-airbnb-room-type-predictor-1-38ff.onrender.com](https://nyc-airbnb-room-type-predictor-1-38ff.onrender.com)

**📊 GitHub Repository:** [github.com/tarunkumar7906/NYC-Airbnb-Room-Type-Predictor](https://github.com/tarunkumar7906/NYC-Airbnb-Room-Type-Predictor)

**📓 Dataset:** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (Kaggle) — 48,895 real listings

---

## 🎯 The Real-World Problem

Airbnb listings need accurate room-type classification for:
- **Search & Filtering** — helping guests find exactly what they want
- **Pricing Fairness** — comparing similar room types within the same market
- **Data Quality Checks** — flagging listings where the pattern of price, reviews, and availability doesn't match the stated room type
- **Market Analysis** — understanding room-type distribution across NYC's five boroughs

This project solves it with a **multi-class classification model**, trained on nearly 49,000 real NYC listings, that predicts room type from 10 features — location, price, minimum stay, and host activity — and serves it through a live web app.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              FRONTEND — "StayType AI"                        │
│         HTML5 + CSS3 (Dark Theme) + Vanilla JavaScript       │
│  - Animated gradient orbs + grid overlay background          │
│  - Cascading borough → neighbourhood dropdown (214 options)  │
│  - Live API health indicator (polls every 20s)                │
│  - Animated probability bars with shimmer effect             │
├──────────────────────────────────────────────────────────────┤
│               FETCH API / HTTPS (JSON)                       │
├──────────────────────────────────────────────────────────────┤
│            BACKEND — FastAPI REST API                        │
│  - Pydantic Field-level validation (ge, le, min_length)      │
│  - Single POST /predict endpoint                              │
│  - CORS enabled for all origins                                │
├──────────────────────────────────────────────────────────────┤
│    ML MODEL — Tuned Random Forest Pipeline (Scikit-Learn)     │
│  - ColumnTransformer: median-impute+scale (numeric),          │
│    most-frequent-impute+one-hot (categorical)                 │
│  - Trained & validated on 48,895 real listings                │
│  - Tuned via RandomizedSearchCV, optimized for macro-F1        │
│  - Test accuracy: 85.5% | Macro-F1: 0.745                      │
├──────────────────────────────────────────────────────────────┤
│           DEPLOYMENT — Render Cloud Platform                  │
│  - Live at nyc-airbnb-room-type-predictor-1-38ff.onrender.com │
│  - Auto-deploy from GitHub, HTTPS, 24/7 uptime                │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 The Dataset

- **Source:** NYC Airbnb Open Data (Kaggle, via `kagglehub`)
- **File:** `AB_NYC_2019.csv`
- **Size:** 48,895 listings × 16 original columns
- **Target:** `room_type` — 3 classes (imbalanced, with **Shared Room** as a small minority class)

This class imbalance is the reason the whole project is built around **macro-F1**, not plain accuracy — accuracy alone can look good on an imbalanced dataset while badly ignoring the minority class.

---

## 🧪 Machine Learning Development Process

The notebook (`nyc_airbnb_room.ipynb`) follows a complete, disciplined ML workflow — 19 markdown cells explaining decisions, 28 code cells implementing them.

### 1. Exploratory Data Analysis
Followed the standard order every ML engineer should follow:
- **Missing values** check
- **Univariate analysis** — distributions and skew of numeric features; frequency of categorical features
- **Bivariate analysis** — how each feature relates to `room_type`
- **Correlation** between numeric features
- **Geographic distribution** — plotting listings by latitude/longitude as a bonus visual, confirming that location clusters align with borough boundaries

**Key EDA finding:** The target classes are imbalanced — **Shared Room is a small minority** compared to Entire home/apt and Private room. This finding directly shaped every later modeling decision (stratified splits, `class_weight='balanced'`, macro-F1 scoring).

### 2. Data Cleaning & Feature Engineering
1. **Dropped identifier/free-text columns** with no generalizable signal for a tabular model: `id`, `name`, `host_id`, `host_name`, `last_review`
2. **Filled missing `reviews_per_month` with 0** — reasoning: no reviews yet means zero, not missing data
3. **Capped outliers instead of deleting rows** — clipped `price` and `minimum_nights` at their 99th percentile, so a handful of data-entry errors (e.g. a $10,000/night listing, or 1,250 minimum nights) don't distort the model without throwing away real listings
4. **Separated features (X) from target (y)**

### 3. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)
```
- **67% train / 33% test**
- **`stratify=y`** — critical because the target is imbalanced; this guarantees the same class proportions appear in both the training and test sets
- The test set is touched **exactly once**, at the very end, to report final honest performance

### 4. Preprocessing Pipeline (ColumnTransformer)
Built once, reused everywhere, to avoid data leakage:

| Column Type | Columns | Steps |
|-------------|---------|-------|
| **Numeric** | latitude, longitude, price, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365 | Median imputation → StandardScaler |
| **Categorical** | neighbourhood_group, neighbourhood | Most-frequent imputation → OneHotEncoder (`handle_unknown='ignore'`) |

**Why this matters:** Preprocessing is learned *only* on training data and applied consistently everywhere — no leakage from test data into training.

### 5. Model Comparison (Fair, Cross-Validated)
Four algorithms, each wrapped in the *same* preprocessing pipeline, evaluated with **3-fold stratified cross-validation** on the training set only:

| Model | CV Accuracy | CV Macro-F1 |
|-------|-------------|-------------|
| Logistic Regression | 0.659 | 0.522 |
| Decision Tree | 0.782 | 0.648 |
| **Random Forest** | **0.849** | **0.734** |
| Gradient Boosting | 0.850 | 0.705 |

**Selected: Random Forest** — Gradient Boosting had marginally higher accuracy (0.850 vs 0.849), but Random Forest had a **notably better macro-F1 (0.734 vs 0.705)**, meaning it handled the minority "Shared Room" class more fairly. On an imbalanced problem, that trade-off matters more than the last decimal point of accuracy.

### 6. Hyperparameter Tuning (RandomizedSearchCV)
```python
param_distribution = {
    "classifier__n_estimators": [100, 200, 150, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
search = RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=param_distribution,
    n_iter=10, cv=3, scoring="f1_macro", random_state=42
)
```
- **10 random parameter combinations**, 3-fold CV, optimizing for **macro-F1** (not accuracy) because of the class imbalance
- **Best parameters found:** `n_estimators=150`, `min_samples_split=5`, `max_depth=None`
- **Best CV macro-F1:** 0.734

### 7. Final Evaluation (Held-Out Test Set — Touched Once)
```
Accuracy Score: 0.8547  (85.5%)
F1 Score (macro): 0.7452
```
Plus a confusion matrix heatmap to see exactly where the model confuses classes (most confusion happens between Private room and Shared room — the two hardest classes to tell apart, which lines up with real-world intuition).

### 8. Model Serialization
```python
joblib.dump(best_pipeline, "Model_Pipeline.pkl")
```
The **entire pipeline** — preprocessing and model together — is saved as a single artifact. In production, a new listing is scored with **one call**, no manual preprocessing required at inference time.

---

## 📈 Final Model Performance

| Metric | Cross-Validation | Held-Out Test Set |
|--------|-------------------|---------------------|
| **Accuracy** | 84.9% | **85.5%** |
| **Macro-F1** | 0.734 | **0.745** |

**What this means:**
- The model correctly classifies room type for **~85 out of 100** unseen listings
- **Macro-F1 of 0.745** confirms the model isn't just good at the majority classes — it performs reasonably across all three room types, including the minority Shared Room class
- Test performance is *consistent* with (slightly better than) cross-validation performance — a good sign there's no overfitting to the training folds

---

## 🛠️ Tech Stack

### Machine Learning
- **Language:** Python (Pandas, NumPy)
- **Data Source:** `kagglehub` (direct Kaggle dataset download)
- **Preprocessing:** Scikit-learn `ColumnTransformer` + `Pipeline`
- **Models Compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Final Model:** Random Forest Classifier (tuned)
- **Tuning:** `RandomizedSearchCV` (10 iterations, 3-fold CV, macro-F1 scoring)
- **Serialization:** joblib → `Model_Pipeline.pkl`

### Backend (45 lines Python)
- **Framework:** FastAPI
- **Validation:** Pydantic `Field()` constraints matching the training data's valid ranges exactly
- **Prediction Logic:**
  ```python
  row = pd.DataFrame([features.dict()], columns=COLUMNS)
  prediction = model.predict(row)
  probability = model.predict_proba(row)
  ```
- **Endpoints:** `GET /` (health check) · `POST /predict` (classification + probabilities)
- **CORS:** Enabled for all origins

### Frontend (205 HTML + 412 CSS + 398 JavaScript = 1,015 lines)
- **Design:** Dark theme with animated gradient orbs, glassmorphism panels, grid overlay
- **Cascading Dropdowns:** Selecting a borough dynamically populates its neighbourhoods — hardcoded to match the **exact 214 neighbourhood categories** the OneHotEncoder was trained on, so users can never submit an unseen category
- **Sample Data Generator:** One-click "Use a sample NYC location" button fills realistic test data using real borough centroid coordinates
- **Live API Health Check:** Polls the backend every 20 seconds, shows online/offline status
- **Animated Confidence Bars:** Shows the full 3-class probability breakdown, not just the top prediction, with staggered animation and shimmer effect on the winning class
- **Configurable API Endpoint:** Saved to `localStorage`, so developers can point the same frontend at a local backend with zero code changes

### Deployment
- **Platform:** Render
- **Model Storage:** Git LFS (Large File Storage) for the `.pkl` file
- **Dependencies:** FastAPI ≥0.110, Uvicorn ≥0.29, Pandas ≥2.0, Scikit-learn ≥1.9, Joblib ≥1.3, Pydantic ≥2.0

---

## 📋 Input Features (10 Parameters)

| Feature | Type | Validation | Description |
|---------|------|-----------|--------------|
| **latitude** | Float | -90 to 90 | Listing's latitude |
| **longitude** | Float | -180 to 180 | Listing's longitude |
| **price** | Float | ≥ 0 | Price per night (USD), capped at 99th percentile in training |
| **minimum_nights** | Integer | 1 to 365 | Minimum nights required to book |
| **number_of_reviews** | Integer | ≥ 0 | Total number of reviews received |
| **reviews_per_month** | Float | ≥ 0 | Average monthly review rate (0 if no reviews yet) |
| **calculated_host_listings_count** | Integer | ≥ 0 | Number of listings this host manages |
| **availability_365** | Integer | 0 to 365 | Days available for booking per year |
| **neighbourhood_group** | String | Required | NYC borough |
| **neighbourhood** | String | Required | Specific neighbourhood (1 of 214, matched to training categories) |

---

## 📊 Output

**Predicted Room Type** — one of 3 classes:
- 🏠 **Entire home/apt**
- 🛏️ **Private room**
- 🛋️ **Shared room** (minority class — model still trained to recognize it fairly via macro-F1 optimization)

**Plus:** Full probability distribution across all three classes.

**Example Response:**
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.72, 0.24, 0.04]
}
```

---

## 🌆 NYC Borough & Neighbourhood Coverage

| Borough | Neighbourhoods |
|---------|-----------------|
| **Manhattan** | 30 |
| **Brooklyn** | 47 |
| **Queens** | 51 |
| **Bronx** | 48 |
| **Staten Island** | 38 |

**Total: 214 neighbourhoods**, hardcoded in the frontend to exactly match what the `OneHotEncoder` saw during training — guaranteeing every submission is a category the model actually understands.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- Git + Git LFS (the model file is tracked with LFS)
- Modern web browser

### Local Installation

#### 1. Clone the Repository
```bash
git lfs install
git clone https://github.com/tarunkumar7906/NYC-Airbnb-Room-Type-Predictor.git
cd NYC-Airbnb-Room-Type-Predictor
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Backend API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 5. Open the Frontend
```bash
python -m http.server 5500
# Visit http://localhost:5500
```

#### 6. Point Frontend to Local API
Use the **API URL field in the footer** to switch from the deployed Render URL to `http://127.0.0.1:8000` — no code editing needed, it's saved to `localStorage`.

### Re-Running the Notebook
```bash
pip install kagglehub pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook nyc_airbnb_room.ipynb
```
The notebook downloads the dataset automatically via `kagglehub` — no manual CSV download required.

---

## 📡 API Documentation

### Base URL
```
Production: https://nyc-airbnb-room-type-predictor-1-38ff.onrender.com
Local:      http://localhost:8000
```

### GET `/`
Health check → returns `"Hello Guys"`

### POST `/predict`

**Request Body:**
```json
{
  "latitude": 40.7128,
  "longitude": -73.9352,
  "price": 150,
  "minimum_nights": 3,
  "number_of_reviews": 24,
  "reviews_per_month": 1.4,
  "calculated_host_listings_count": 1,
  "availability_365": 200,
  "neighbourhood_group": "Brooklyn",
  "neighbourhood": "Williamsburg"
}
```

**Success Response (200 OK):**
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.72, 0.24, 0.04]
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "price"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## 🎨 Frontend UX Highlights

**"Use a Sample NYC Location" Button** — randomly picks a borough, jitters coordinates around its real centroid, auto-selects a valid neighbourhood, and fills realistic price/review/availability values. Makes the app instantly demoable.

**Live API Status Indicator** — 🟡 Checking → 🟢 Online (pulsing) → 🔴 Offline, polling every 20 seconds. Useful since Render's free tier spins down when idle.

**Animated Confidence Breakdown** — shows all three class probabilities as animated bars, sorted by confidence, with the top prediction highlighted in a distinct gradient with a shimmer effect.

---

## 📁 Project Structure

```
NYC-Airbnb-Room-Type-Predictor/
│
├── main.py                       # FastAPI backend (45 lines)
├── Model_Pipeline.pkl             # Tuned Random Forest pipeline (Git LFS)
├── requirements.txt               # Python dependencies
│
├── index.html                     # Frontend markup (205 lines)
├── style.css                      # Dark theme styling (412 lines)
├── script.js                      # Frontend logic (398 lines)
│
├── nyc_airbnb_room.ipynb          # Complete ML notebook
│                                  # - 19 markdown cells (explanations)
│                                  # - 28 code cells (implementation)
│                                  # - EDA → Cleaning → Preprocessing →
│                                  #   Model comparison → Tuning → Evaluation
│
├── .gitattributes                 # Git LFS config for the .pkl model
├── .gitignore
└── README.md
```

---

## 💡 Key Engineering Decisions

**1. Macro-F1 over Accuracy for Model Selection**
Because Shared Room is a small minority class, plain accuracy could look artificially high while the model ignores that class entirely. Macro-F1 treats all three classes equally, forcing the model selection and tuning process to actually care about the minority class.

**2. Outlier Capping Instead of Row Deletion**
Extreme values in `price` (e.g. $10,000/night) and `minimum_nights` (e.g. 1,250 nights) are almost certainly data-entry errors. Rather than deleting those listings (losing potentially valid signal in their other features), the notebook **clips** these two columns at their 99th percentile — reducing distortion without discarding data.

**3. Stratified Splitting Throughout**
Both the train/test split and the cross-validation folds use `stratify=y` / `StratifiedKFold`-style behavior, guaranteeing the rare Shared Room class is proportionally represented in every fold and split — not just clumped into one.

**4. Same Preprocessor, Every Model**
All four candidate models (Logistic Regression → Gradient Boosting) are wrapped in the exact same `ColumnTransformer`, so the comparison between them is a fair test of the *algorithm*, not an artifact of different preprocessing.

**5. Hardcoded Neighbourhood List in the Frontend**
Rather than letting users type any neighbourhood name freely, the frontend embeds the exact 214 neighbourhoods the model saw during training. This prevents `OneHotEncoder(handle_unknown='ignore')` from silently zeroing-out an unrecognized category and returning a degraded prediction.

---

## 🔒 Security & Validation

✅ Pydantic `Field()` constraints matching training data's valid ranges  
✅ String length validation on borough/neighbourhood  
✅ CORS configured for frontend-backend communication  
✅ No data persistence — predictions are fully stateless  
✅ Defense in depth — client-side JS validation + server-side Pydantic validation

---

## 🚀 Future Improvements

- [ ] Add SHAP or feature-importance visualization to explain individual predictions
- [ ] Add a map picker (Leaflet/Mapbox) instead of typing lat/long manually
- [ ] Batch prediction endpoint (CSV upload → predictions for multiple listings)
- [ ] Re-train periodically as NYC Open Data releases new snapshots
- [ ] Try boosting libraries (XGBoost, LightGBM, CatBoost) beyond sklearn's Gradient Boosting
- [ ] Address Shared Room minority class further with SMOTE or targeted oversampling
- [ ] Add price recommendation as a companion prediction

---

## 🎓 What This Project Demonstrates

✅ **Honest, imbalance-aware ML** — choosing macro-F1 over accuracy, and picking the *fairer* model over the marginally more accurate one  
✅ **No data leakage** — single reusable preprocessing pipeline, test set touched exactly once  
✅ **Fair model comparison** — 4 algorithms, same preprocessor, same CV folds  
✅ **Real-world data cleaning** — capping instead of deleting outliers, thoughtful missing-value handling  
✅ **Production ML serving** — FastAPI + Pydantic + a single serialized pipeline artifact  
✅ **Thoughtful frontend engineering** — category-safe dropdowns, live health checks, sample data generation  
✅ **Full deployment** — Git LFS for the model, live on Render, working end-to-end

---

## ⚖️ License

Open-source project available for educational purposes.

---

## 🤝 Contact & Social

**Author:** Tarun Kumar  
**GitHub:** [github.com/tarunkumar7906](https://github.com/tarunkumar7906)  
**LinkedIn:** [linkedin.com/in/tarun-kumar](https://www.linkedin.com/in/tarun-kumar-5b9280396)

---

**Project Status:** ✅ Live & Deployed  
**Model Type:** Multi-Class Classification (Tuned Random Forest)  
**Test Accuracy:** 85.5% | **Test Macro-F1:** 0.745  
**Classes:** Entire home/apt · Private room · Shared room  
**Last Updated:** September 2026
