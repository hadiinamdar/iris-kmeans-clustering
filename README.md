<div align="center">

# 🌸 Iris K-Means Clustering

**Unsupervised clustering of the Iris dataset with an interactive Streamlit dashboard**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[Overview](#-overview) •
[Features](#-features) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Project Structure](#-project-structure) •
[Methodology](#-methodology)

</div>

---

## 📖 Overview

This project applies **K-Means clustering** to the classic Iris dataset (150 samples, 4 measurements, 3 species) and evaluates how well the discovered clusters line up with the real species.

The species labels are used **only for evaluation after clustering**. They are never used to fit the model.

The repository contains:

- 📓 A **Jupyter notebook** with the full analysis, from exploration to evaluation.
- 🖥️ An **interactive Streamlit app** (`app.py`) for exploring the data, comparing values of K and predicting clusters for new flowers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Exploratory analysis** | Data preview, summary statistics, species distribution and optional pairplot |
| 📏 **Feature scaling** | `StandardScaler` applied before clustering |
| 📉 **Elbow method** | Inertia plotted across a configurable range of K |
| 🎯 **Silhouette analysis** | Cluster quality scored for each K |
| ✅ **Accuracy vs. species** | Clusters mapped to their majority species for post-hoc evaluation |
| 🧮 **Confusion matrix** | Shows which species are mixed up |
| 🗺️ **PCA visualization** | 2D projection of the clusters with explained variance |
| 🌼 **Live prediction** | Sliders to enter new measurements and get a cluster |
| 💾 **CSV export** | Download the dataset with cluster labels attached |
| 📂 **Custom data** | Upload your own Iris-format CSV from the sidebar |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 5. Or open the notebook

```bash
jupyter notebook Iris_KMeans_Clustering_Corrected.ipynb
```

---

## 🖥️ Using the App

| Tab | What you can do |
|---|---|
| **Data** | Inspect the dataset, statistics and class balance |
| **Choosing K** | Compare inertia, silhouette score and accuracy across K values |
| **Final model** | View metrics, PCA cluster plot, confusion matrix, and download results |
| **Predict a flower** | Enter sepal and petal measurements to find the matching cluster |

**Sidebar controls:** final K, K range to compare, random seed, optional CSV upload, pairplot toggle.

> 💡 **Tip:** If you upload your own CSV it needs the columns `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm` and `Species`. If no file is provided, the app uses `Iris.csv` when present, otherwise scikit-learn's built-in Iris data.

---

## 📊 Results

Metrics on standardized features (`random_state=42`, `n_init=10`):

| K | Inertia | Silhouette Score | Accuracy vs. Species |
|:-:|--------:|-----------------:|---------------------:|
| 2 | 222.36 | **0.5818** | 66.67% |
| **3** | **139.82** | **0.4599** | **83.33%** |
| 4 | 114.09 | 0.3869 | 83.33% |
| 5 | 90.93 | 0.3459 | 84.67% |
| 6 | 81.54 | 0.3171 | 92.00% |

### Key takeaways

- **K = 3 is the final model** because the dataset contains three known species.
- **K = 2 has the highest silhouette score** because *Iris setosa* is clearly separated, while *versicolor* and *virginica* overlap and tend to merge into one cluster.
- **Accuracy rises with K**, but this is expected: with more clusters, each cluster is purer and the majority-label mapping becomes easier. It does not mean a larger K is a better model.
- Setosa is clustered almost perfectly. Most errors come from the versicolor/virginica overlap.

---

## 🧪 Methodology

```
Load data → Explore → Select features → Standardize → Fit K-Means (K = 2…6)
        → Elbow + Silhouette → Map clusters to majority species
        → Accuracy + Confusion matrix → PCA visualization
```

1. **Features:** sepal length, sepal width, petal length, petal width.
2. **Scaling:** `StandardScaler` so all features contribute equally to distances.
3. **Model:** `KMeans(init="k-means++", n_init=10, random_state=42)`.
4. **Diagnostics:** the elbow method (inertia) and silhouette score, which need no labels.
5. **Evaluation:** each cluster is assigned its majority species, then accuracy and a confusion matrix are computed.
6. **Visualization:** PCA reduces the four features to two components for plotting.

> ⚠️ **Note:** K-Means cluster IDs are arbitrary, so raw cluster numbers can't be compared with species codes directly. Clustering accuracy is a post-hoc agreement measure and is not the same as the unsupervised silhouette score.

---

## 📁 Project Structure

```
.
├── app.py                                  # Streamlit dashboard
├── Iris_KMeans_Clustering_Corrected.ipynb  # Full analysis notebook
├── Iris.csv                                # Optional dataset (falls back to sklearn)
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** for the core language
- **scikit-learn** for K-Means, PCA, scaling and metrics
- **pandas / NumPy / SciPy** for data handling
- **Matplotlib / Seaborn** for visualization
- **Streamlit** for the interactive app

---

## 🔮 Future Improvements

- [ ] Add silhouette plots per cluster
- [ ] Compare against other algorithms (DBSCAN, Agglomerative, GMM)
- [ ] Add 3D PCA visualization
- [ ] Deploy to Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

<div align="center">

**If you found this project useful, please give it a ⭐**

Made with ❤️ using Python and Streamlit

</div>
