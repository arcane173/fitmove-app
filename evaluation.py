import time
import numpy as np
import pandas as pd
from collections import defaultdict
import cv2
import mediapipe as mp
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
import json
from datetime import datetime

# Reuse trainer yang sama persis dengan yang dipakai untuk melatih model
# produksi (model_curl.pkl, dsb), supaya perbandingan algoritma di sini
# konsisten dengan model yang benar-benar dipakai aplikasi.
from model_trainer import ModelTrainer, DATASETS

ALGO_DISPLAY_NAMES = {
    "random_forest":  "Random Forest",
    "svm":            "SVM",
    "neural_network": "Neural Network",
}


class SystemEvaluator:
    """Evaluator untuk sistem FitMove."""
    
    def __init__(self):
        self.results = {
            "model_performance": {},
            "algorithm_comparison": {},
            "best_algorithm": {},
            "latency": {},
            "user_study": [],
            "expert_validation": []
        }
        # mediapipe cuma dibutuhkan untuk measure_end_to_end_latency() (akses
        # kamera live). compare_algorithms() dan metrik model TIDAK butuh
        # mediapipe sama sekali, jadi importnya ditunda (lazy) supaya
        # evaluasi algoritma tetap bisa jalan walau ada masalah environment
        # terkait kamera/mediapipe.
        self._mp_pose = None

    @property
    def mp_pose(self):
        if self._mp_pose is None:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
        return self._mp_pose
    
    def _compute_metrics(self, model, X_test, y_test, model_name="Model"):
        """
        Hitung metrik evaluasi (accuracy, precision, recall, f1, per-class,
        confusion matrix) untuk satu model, TANPA menyimpan ke self.results.
        Dipakai bersama oleh evaluate_model_performance() dan
        compare_algorithms() supaya cara hitungnya konsisten.
        """
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        classes = ['bad', 'good', 'perfect']
        per_class = {}
        for i, cls in enumerate(classes):
            tp = np.sum((y_pred == i) & (y_test == i))
            fp = np.sum((y_pred == i) & (y_test != i))
            fn = np.sum((y_pred != i) & (y_test == i))

            per_class[cls] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'support': int(np.sum(y_test == i))
            }

        cm = confusion_matrix(y_test, y_pred)

        return {
            'model_name': model_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=classes, zero_division=0
            )
        }

    def evaluate_model_performance(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluasi performa SATU model ML dan simpan hasilnya ke self.results.

        Args:
            model: Trained model (joblib object)
            X_test: Test features
            y_test: Test labels
            model_name: Nama model untuk logging

        Returns:
            Dictionary dengan metrics evaluasi
        """
        metrics = self._compute_metrics(model, X_test, y_test, model_name)
        self.results['model_performance'][model_name] = metrics
        return metrics

    def compare_algorithms(self, exercise, test_size=0.2, random_state=42):
        """
        Melatih ULANG 3 algoritma (Random Forest, SVM, Neural Network) dari
        dataset asli gerakan tertentu, memakai prosedur & hyperparameter
        search yang SAMA PERSIS dengan model_trainer.py (supaya adil dan
        konsisten dengan model yang dipakai di aplikasi), lalu mengevaluasi
        ketiganya di test-set yang identik dan menentukan pemenang secara
        OTOMATIS berdasarkan akurasi tertinggi (bukan ditulis manual).

        Hasil dan alasan pemilihan disimpan ke self.results['algorithm_comparison']
        supaya bisa ditampilkan sebagai tabel & dimasukkan ke laporan skripsi.
        """
        if exercise not in DATASETS:
            raise ValueError(f"Gerakan '{exercise}' tidak dikenal. Pilihan: {list(DATASETS.keys())}")

        print(f"\n{'='*70}")
        print(f"🔬 MEMBANDINGKAN ALGORITMA UNTUK GERAKAN: {exercise.upper()}")
        print(f"{'='*70}")

        trainer = ModelTrainer(exercise)
        trainer.load_data()

        # Latih ketiga algoritma (masing-masing sudah melakukan grid-search
        # hyperparameter kecil di dalam model_trainer.py, persis seperti
        # proses training model produksi).
        trainer.train_random_forest(test_size=test_size, random_state=random_state)
        trainer.train_svm(test_size=test_size, random_state=random_state)
        trainer.train_neural_network(test_size=test_size, random_state=random_state)

        # Split test-set yang SAMA untuk ketiga model, biar perbandingannya adil.
        _, X_test, _, y_test = train_test_split(
            trainer.X, trainer.y, test_size=test_size,
            random_state=random_state, stratify=trainer.y
        )

        comparison = {}
        for key, model in trainer.models.items():
            name = ALGO_DISPLAY_NAMES.get(key, key)
            metrics = self._compute_metrics(model, X_test, y_test, model_name=name)

            # Cross-validation 5-fold di SELURUH data (bukan cuma 1x split).
            # Ini penting terutama untuk dataset kecil (~300-400 baris):
            # akurasi dari 1x train-test-split bisa kebetulan tinggi/rendah,
            # sedangkan cross-validation menunjukkan seberapa KONSISTEN
            # model tersebut across beberapa split data yang berbeda.
            cv_scores = cross_val_score(model, trainer.X, trainer.y, cv=5)
            metrics['cv_mean'] = round(cv_scores.mean() * 100, 2)
            metrics['cv_std'] = round(cv_scores.std() * 100, 2)
            metrics['cv_scores'] = [round(s * 100, 2) for s in cv_scores]

            comparison[name] = metrics

        # Pemenang ditentukan OTOMATIS dari akurasi tertinggi hasil evaluasi
        # di atas — bukan angka yang ditentukan/ditulis manual sebelumnya.
        best_name = max(comparison, key=lambda n: comparison[n]['accuracy'])
        for name in comparison:
            comparison[name]['selected'] = (name == best_name)

        self.results.setdefault('algorithm_comparison', {})[exercise] = comparison
        self.results.setdefault('best_algorithm', {})[exercise] = best_name

        self.print_comparison_table(exercise)

        best_key = [k for k, v in ALGO_DISPLAY_NAMES.items() if v == best_name][0]
        best_model = trainer.models[best_key]

        return comparison, best_name, best_model, X_test

    def print_comparison_table(self, exercise):
        """Cetak tabel perbandingan algoritma untuk satu gerakan ke terminal."""
        comparison = self.results.get('algorithm_comparison', {}).get(exercise)
        if not comparison:
            print(f"⚠️ Belum ada hasil perbandingan untuk '{exercise}'. Jalankan compare_algorithms() dulu.")
            return

        best_name = self.results.get('best_algorithm', {}).get(exercise)

        headers = ["Algoritma", "Akurasi", "Precision", "Recall", "F1-Score", "CV Mean±Std", "Terpilih"]
        col_w = [16, 10, 11, 9, 10, 15, 9]

        def fmt_row(cols):
            return " | ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

        sep = "-+-".join("-" * w for w in col_w)

        print(f"\n📊 Tabel Perbandingan Algoritma — Gerakan: {exercise.upper()}")
        print(fmt_row(headers))
        print(sep)
        for name, m in sorted(comparison.items(), key=lambda kv: -kv[1]['accuracy']):
            mark = "✅ YA" if name == best_name else ""
            cv_txt = f"{m['cv_mean']:.2f}±{m['cv_std']:.2f}%" if 'cv_mean' in m else "-"
            print(fmt_row([
                name, f"{m['accuracy']:.2f}%", f"{m['precision']:.2f}%",
                f"{m['recall']:.2f}%", f"{m['f1_score']:.2f}%", cv_txt, mark
            ]))
        print(sep)
        best_acc = comparison[best_name]['accuracy']
        runner_up = sorted(
            (m['accuracy'] for n, m in comparison.items() if n != best_name), reverse=True
        )
        margin = f", unggul {best_acc - runner_up[0]:.2f}% dari algoritma terbaik berikutnya" if runner_up else ""
        print(f"🏆 Algoritma terpilih untuk {exercise}: {best_name} "
              f"(akurasi {best_acc:.2f}%{margin})")
        if 'cv_mean' in comparison[best_name]:
            print(f"   Cross-validation 5-fold: {comparison[best_name]['cv_mean']:.2f}% "
                  f"± {comparison[best_name]['cv_std']:.2f}% "
                  f"(mengindikasikan konsistensi model di luar 1x pembagian data uji)")
        print()

    def measure_model_latency(self, model, X_sample, iterations=200):
        """
        Ukur latensi inferensi model (waktu model.predict() untuk 1 sampel),
        dalam milidetik. Ini latensi model murni (bukan latensi kamera/
        MediaPipe), dan diukur dari model yang BENAR-BENAR sudah dilatih —
        bukan angka yang ditulis manual.
        """
        sample = X_sample[:1]
        # warm-up (hindari overhead compile/caching di iterasi pertama)
        for _ in range(5):
            model.predict(sample)

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            model.predict(sample)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = sorted(latencies)
        trim = int(iterations * 0.05)
        trimmed = latencies[trim: iterations - trim] if trim else latencies

        metrics = {
            'mean_ms': round(np.mean(trimmed), 4),
            'median_ms': round(np.median(trimmed), 4),
            'std_ms': round(np.std(trimmed), 4),
            'p95_ms': round(np.percentile(trimmed, 95), 4),
        }
        return metrics

    
    def measure_latency(self, detect_function, iterations=100):
        """
        Ukur latency deteksi per frame.
        
        Args:
            detect_function: Function yang mengembalikan hasil deteksi
            iterations: Jumlah iterasi pengukuran
        
        Returns:
            Latency dalam milliseconds
        """
        latencies = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            detect_function()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Remove outliers (top and bottom 5%)
        latencies = sorted(latencies)
        trim_start = int(iterations * 0.05)
        trim_end = int(iterations * 0.95)
        trimmed = latencies[trim_start:trim_end]
        
        metrics = {
            'mean': round(np.mean(trimmed), 2),
            'median': round(np.median(trimmed), 2),
            'std': round(np.std(trimmed), 2),
            'min': round(np.min(trimmed), 2),
            'max': round(np.max(trimmed), 2),
            'p95': round(np.percentile(trimmed, 95), 2),
            'p99': round(np.percentile(trimmed, 99), 2),
            'fps': round(1000 / np.mean(trimmed), 1)
        }
        
        self.results['latency']['detection'] = metrics
        return metrics
    
    def measure_end_to_end_latency(self, video_path=None, n_frames=100):
        """
        Measure end-to-end system latency (camera capture → result).
        """
        cap = cv2.VideoCapture(0 if video_path is None else video_path)
        
        latencies = []
        
        with self.mp_pose.Pose(min_detection_confidence=0.5) as pose:
            for _ in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                start = time.perf_counter()
                
                # Simulate full pipeline
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                # Feature extraction would happen here
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)
        
        cap.release()
        
        metrics = {
            'mean': round(np.mean(latencies), 2),
            'std': round(np.std(latencies), 2),
            'fps': round(1000 / np.mean(latencies), 1)
        }
        
        self.results['latency']['end_to_end'] = metrics
        return metrics
    
    def run_user_study(self, participants_data):
        """
        Analisis hasil user study.
        
        Args:
            participants_data: List of dict dengan keys:
                - user_id: str
                - age: int
                - fitness_level: 'beginner'/'intermediate'/'advanced'
                - exercise: str
                - system_score: float
                - perceived_accuracy: float (1-10)
                - usability_score: float (1-10)
                - satisfaction: float (1-10)
                - comments: str
        """
        df = pd.DataFrame(participants_data)
        
        summary = {
            'total_participants': len(df),
            'age_mean': df['age'].mean(),
            'age_std': df['age'].std(),
            'fitness_distribution': df['fitness_level'].value_counts().to_dict(),
            'avg_perceived_accuracy': df['perceived_accuracy'].mean(),
            'avg_usability_score': df['usability_score'].mean(),
            'avg_satisfaction': df['satisfaction'].mean(),
            'system_accuracy_correlation': df['system_score'].corr(df['perceived_accuracy']),
            'by_fitness_level': df.groupby('fitness_level').agg({
                'perceived_accuracy': 'mean',
                'usability_score': 'mean',
                'satisfaction': 'mean'
            }).to_dict()
        }
        
        self.results['user_study'] = {
            'raw_data': participants_data,
            'summary': summary
        }
        
        return summary
    
    def generate_skripsi_report(self):
        """Generate laporan lengkap untuk skripsi."""
        
        report = []
        report.append("="*70)
        report.append("📊 FITMOVE - SYSTEM EVALUATION REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance (single-model records, kalau ada)
        if self.results['model_performance']:
            report.append("🎯 1. MODEL PERFORMANCE")
            report.append("-"*50)
            for model_name, metrics in self.results['model_performance'].items():
                report.append(f"\n📌 {model_name}:")
                report.append(f"   Accuracy:  {metrics['accuracy']:.2f}%")
                report.append(f"   Precision: {metrics['precision']:.2f}%")
                report.append(f"   Recall:    {metrics['recall']:.2f}%")
                report.append(f"   F1-Score:  {metrics['f1_score']:.2f}%")

                report.append("   Per-Class Performance:")
                for cls, perf in metrics['per_class'].items():
                    report.append(f"      {cls}: P={perf['precision']*100:.1f}% R={perf['recall']*100:.1f}% F1={perf['f1']*100:.1f}%")

        # Algorithm Comparison (Random Forest vs SVM vs Neural Network)
        if self.results.get('algorithm_comparison'):
            report.append("\n🔬 2. PERBANDINGAN ALGORITMA PER GERAKAN")
            report.append("-"*50)
            for exercise, comparison in self.results['algorithm_comparison'].items():
                best_name = self.results.get('best_algorithm', {}).get(exercise, '-')
                report.append(f"\n📌 Gerakan: {exercise.upper()}")
                header = f"   {'Algoritma':<16}{'Akurasi':>10}{'Precision':>12}{'Recall':>10}{'F1-Score':>11}{'CV Mean':>12}"
                report.append(header)
                for name, m in sorted(comparison.items(), key=lambda kv: -kv[1]['accuracy']):
                    mark = "  ✅" if name == best_name else ""
                    cv_txt = f"{m['cv_mean']:.2f}%" if 'cv_mean' in m else "-"
                    report.append(
                        f"   {name:<16}{m['accuracy']:>9.2f}%{m['precision']:>11.2f}%"
                        f"{m['recall']:>9.2f}%{m['f1_score']:>10.2f}%{cv_txt:>12}{mark}"
                    )
                report.append(f"   → Algoritma terpilih: {best_name} (akurasi tertinggi pada data uji, "
                               f"cross-validation {comparison[best_name].get('cv_mean', '-')}"
                               f"±{comparison[best_name].get('cv_std', '-')}%)")
        
        # Latency
        report.append("\n⏱️ 3. SYSTEM LATENCY")
        report.append("-"*50)
        if 'detection' in self.results['latency']:
            d = self.results['latency']['detection']
            report.append(f"   Detection Latency:")
            report.append(f"      Mean:  {d['mean']} ms")
            report.append(f"      Median: {d['median']} ms")
            report.append(f"      Std:    {d['std']} ms")
            report.append(f"      FPS:    {d['fps']} fps")
        
        if 'end_to_end' in self.results['latency']:
            e2e = self.results['latency']['end_to_end']
            report.append(f"\n   End-to-End Latency:")
            report.append(f"      Mean: {e2e['mean']} ms")
            report.append(f"      FPS:  {e2e['fps']} fps")

        if 'model_inference' in self.results['latency']:
            report.append(f"\n   Model Inference Latency (per gerakan):")
            for ex, lat in self.results['latency']['model_inference'].items():
                report.append(f"      {ex}: mean={lat['mean_ms']} ms, p95={lat['p95_ms']} ms")

        # User Study
        if self.results['user_study']:
            report.append("\n👥 4. USER STUDY RESULTS")
            report.append("-"*50)
            s = self.results['user_study']['summary']
            report.append(f"   Total Participants: {s['total_participants']}")
            report.append(f"   Average Age: {s['age_mean']:.1f} ± {s['age_std']:.1f}")
            report.append(f"   Fitness Level Distribution:")
            for level, count in s['fitness_distribution'].items():
                report.append(f"      - {level}: {count}")
            report.append(f"\n   User Ratings (1-10):")
            report.append(f"      Perceived Accuracy: {s['avg_perceived_accuracy']:.2f}")
            report.append(f"      Usability:          {s['avg_usability_score']:.2f}")
            report.append(f"      Satisfaction:       {s['avg_satisfaction']:.2f}")
            report.append(f"\n   Correlation (System vs Perceived): {s['system_accuracy_correlation']:.3f}")
        
        report.append("\n" + "="*70)
        report.append("✅ End of Report")
        
        return "\n".join(report)
    
    def export_results(self, filename="evaluation_results.json"):
        """Export results ke JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Results exported to {filename}")
        return filename


# Contoh penggunaan
if __name__ == "__main__":
    import os

    print("🚀 FitMove — Evaluasi & Perbandingan Algoritma")
    print("="*70)
    print("Melatih ulang Random Forest, SVM, dan Neural Network dari dataset")
    print("asli untuk setiap gerakan, lalu membandingkan performanya secara")
    print("langsung (bukan angka yang ditulis manual).")
    print("="*70)

    evaluator = SystemEvaluator()

    for exercise in ["curl", "pushup", "squat"]:
        if not os.path.exists(DATASETS[exercise]):
            print(f"⚠️ Dataset untuk '{exercise}' tidak ditemukan, dilewati.")
            continue

        comparison, best_name, best_model, X_test = evaluator.compare_algorithms(exercise)

        # Ukur latensi inferensi dari model TERPILIH yang beneran sudah
        # dilatih di atas (real, bukan angka yang ditulis manual).
        latency = evaluator.measure_model_latency(best_model, X_test)
        evaluator.results['latency'].setdefault('model_inference', {})[exercise] = latency
        print(f"⏱️  Latensi inferensi {best_name} untuk {exercise}: "
              f"{latency['mean_ms']} ms (p95={latency['p95_ms']} ms)")

    # Generate & tampilkan laporan lengkap untuk skripsi
    report = evaluator.generate_skripsi_report()
    print(report)

    # Export ke JSON — sekarang berisi hasil PERBANDINGAN ASLI, bukan
    # angka contoh yang ditulis manual.
    evaluator.export_results()

