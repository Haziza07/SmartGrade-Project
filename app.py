from flask import Flask, request, render_template
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# Load model yang sudah dilatih
model = joblib.load("prediksi_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ambil input dari form
        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        sleep_hours = float(request.form["sleep_hours"])
        sample_papers = float(request.form["sample_papers"])
        extracurricular = int(request.form["extracurricular"])

        # Hitung kolom tambahan (Practice_Effect)
        practice_effect = hours_studied * sample_papers

        # Buat array input
        X = np.array([[hours_studied, previous_scores, sleep_hours,
                    sample_papers, extracurricular, practice_effect]])

        # Standarisasi Input
        X_scaled = scaler.transform(X)

        # Prediksi performance index
        y_pred = model.predict(X_scaled)[0]

        # ========== ANALISIS PER FAKTOR ==========
        analisis = []

        # Jam belajar
        if hours_studied < 1:
            analisis.append("Jam belajar kamu masih sangat sedikit. Coba tambah durasi belajar secara bertahap.")
        elif 1 <= hours_studied <= 3:
            analisis.append("Jam belajar kamu berada di rentang ideal untuk pembelajaran efektif.")
        elif 3 < hours_studied <= 5:
            analisis.append("Jam belajar kamu cukup baik, tetapi jangan lupa beristirahat agar tetap fokus.")
        else:
            analisis.append("Jam belajar kamu sangat tinggi. Waspadai kelelahan, karena efektivitas bisa menurun setelah 5 jam.")

        # Tidur (versi baru: optimal 6–8 jam)
        if sleep_hours < 6:
            analisis.append("Tidur kamu kurang dari 6 jam — ini bisa menurunkan konsentrasi dan daya ingat.")
        elif 6 <= sleep_hours <= 8:
            analisis.append("Tidur kamu berada di rentang optimal (6–8 jam). Ini sangat baik untuk kinerja akademik!")
        else:
            analisis.append("Tidur lebih dari 8 jam bisa mengurangi waktu produktifmu. Cobalah menjaga pola tidur yang seimbang.")

        # Latihan soal
        if sample_papers < 10:
            analisis.append("Kamu masih jarang berlatih soal. Latihan soal rutin bisa memperkuat pemahaman konsep.")
        elif 10 <= sample_papers <= 30:
            analisis.append("Kamu cukup aktif berlatih soal — bagus! Terus evaluasi kesalahan untuk hasil maksimal.")
        else:
            analisis.append("Kamu sangat rajin berlatih soal! Pastikan juga waktu istirahat cukup agar otak tidak jenuh.")

        # Ekskul
        if extracurricular == 1:
            analisis.append("Aktif dalam kegiatan ekstrakurikuler membantu melatih soft skills dan manajemen waktu.")
        else:
            analisis.append("Pertimbangkan ikut kegiatan ekstrakurikuler agar keseimbangan akademik dan sosial tetap terjaga.")

        # Nilai sebelumnya
        if previous_scores < 60:
            analisis.append("Nilai sebelumnya masih rendah, tapi usaha konsisten bisa meningkatkan performa dengan cepat.")
        elif 60 <= previous_scores < 80:
            analisis.append("Kamu berada di jalur yang baik, teruskan kebiasaan belajarmu dan tingkatkan latihan.")
        else:
            analisis.append("Kamu punya fondasi akademik yang kuat — pertahankan kualitas dan konsistensimu!")

        # ========== Kesimpulan Umum ==========
        if y_pred >= 85:
            kesimpulan = "🎯 Kinerja kamu luar biasa! Kamu menunjukkan keseimbangan belajar, istirahat, dan latihan yang ideal."
            solusi = "Pertahankan pola belajar dan gaya hidupmu. Kamu bisa mulai menantang diri dengan target baru!"
        elif y_pred >= 70:
            kesimpulan = "💪 Kinerja kamu baik! Hanya butuh sedikit penyempurnaan untuk mencapai level terbaik."
            solusi = "Tetap semangat! pastikan juga tidurmu cukup untuk daya fokus yang lebih baik."
        else:
            kesimpulan = "📈 Masih ada ruang untuk berkembang."
            solusi = "Coba atur ulang jadwal belajarmu, sesekali cobalah untuk menarik nafas sejenak. Karena kesehatan itu penting dan mahal"

        # ========== Buat Grafik ==========
        plt.figure(figsize=(5,3))
        plt.bar(["Prediksi Kinerja"], [y_pred], color="cornflowerblue")
        plt.ylim(0, 100)
        plt.title("Prediksi Indeks Kinerja Siswa")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        return render_template("index.html",
                                prediction=round(y_pred, 2),
                                img_data=img_base64,
                                kesimpulan=kesimpulan,
                                solusi=solusi,
                                analisis=analisis)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)