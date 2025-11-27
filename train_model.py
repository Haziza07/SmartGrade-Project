import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("Student_Performance.csv")

# Normalisasi jumlah soal agar tetap realistis (maksimal 50)
df["Sample Question Papers Practiced"] = np.clip(df["Sample Question Papers Practiced"], 0, 50)

# ==========================
# 2. Preprocessing Awal
# ==========================
encoder = LabelEncoder()
df["Extracurricular Activities"] = encoder.fit_transform(df["Extracurricular Activities"])

# Tambahan efek gabungan jam belajar dan latihan soal
df["Practice_Effect"] = df["Hours Studied"] * df["Sample Question Papers Practiced"]

# ==========================
# 3. Penyesuaian Berdasarkan Penelitian
# ==========================

# --- Efek jam belajar non-linear (efektif di 1–4 jam, lalu menurun perlahan) ---
study_bonus = np.where(
    df["Hours Studied"] <= 3,
    1 + 0.08 * df["Hours Studied"],
    1.24 - 0.02 * (df["Hours Studied"] - 3)
)
study_bonus = np.clip(study_bonus, 0.8, 1.3)

# --- Efek tidur (optimal di 6–8 jam, puncak sekitar 7 jam) ---
sleep_effect = np.exp(-((df["Sleep Hours"] - 7) ** 2) / 4) + 0.8
sleep_effect = np.clip(sleep_effect, 0.8, 1.1)

# --- Efek latihan soal (testing effect) ---
practice_effect = 1 + 0.01 * np.sqrt(df["Sample Question Papers Practiced"])
practice_effect = np.clip(practice_effect, 1.0, 1.25)

# --- Efek ekstrakurikuler (sedikit positif) ---
extracurricular_effect = np.where(df["Extracurricular Activities"] == 1, 1.05, 1.0)

# ==========================
# 4. Bangun ulang Performance Index (lebih ilmiah)
# ==========================
df["Performance Index"] = (
    df["Previous Scores"] * 0.55 +
    (df["Hours Studied"] * 7) * 0.15 +
    (df["Sample Question Papers Practiced"] * 0.6) * 0.15 +
    (df["Sleep Hours"] * 10) * 0.10 +
    (df["Extracurricular Activities"] * 5)
)

# Terapkan efek gabungan
df["Performance Index"] *= study_bonus * sleep_effect * practice_effect * extracurricular_effect
df["Performance Index"] = np.clip(df["Performance Index"], 0, 100)

# ==========================
# 5. Fitur dan Target
# ==========================
X = df[[
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Extracurricular Activities",
    "Practice_Effect"
]]
y = df["Performance Index"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 6. Standardisasi
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 7. Training Model
# ==========================
model = RandomForestRegressor(
    random_state=42,
    n_estimators=80,
    max_depth=12,
    min_samples_split=3
)
model.fit(X_train_scaled, y_train)

# ==========================
# 8. Evaluasi
# ==========================
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model dilatih dengan akurasi R²: {r2:.3f}, RMSE: {rmse:.3f}")

# ==========================
# 9. Simpan Model
# ==========================
joblib.dump(model, "prediksi_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)
joblib.dump(encoder, "encoder.pkl", compress=3)
print("✅ Model, scaler, dan encoder berhasil disimpan dan sudah diperkecil!")

# ==========================
# 10. Tes Prediksi Cepat
# ==========================
test_data = {
    "Hours Studied": 3,
    "Previous Scores": 85,
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 20,
    "Extracurricular Activities": 1
}
test_data["Practice_Effect"] = test_data["Hours Studied"] * test_data["Sample Question Papers Practiced"]

predicted = model.predict(scaler.transform(pd.DataFrame([test_data])))[0]
print(f"🔍 Contoh Prediksi Otomatis = {predicted:.2f}")