import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from datetime import datetime

# Konfigurasi
DATASETS = {
    "curl": "dataset_curl.csv",
    "pushup": "dataset_pushup.csv",
    "squat": "dataset_squat.csv"
}

FEATURE_NAMES = {
    "curl": ["left_elbow", "right_elbow", "symmetry", "shoulder_diff", "wrist_height", "torso_lean"],
    "pushup": ["left_elbow", "right_elbow", "body_angle", "symmetry", "hip_deviation", "hand_width"],
    "squat": ["knee_angle", "hip_angle", "knee_cave", "depth_ratio", "symmetry", "torso_lean"]
}

LABEL_MAP = {"bad": 0, "good": 1, "perfect": 2}


class ModelTrainer:
    """Trainer untuk model klasifikasi gerakan workout."""
    
    def __init__(self, exercise):
        self.exercise = exercise
        self.dataset_path = DATASETS.get(exercise)
        self.feature_names = FEATURE_NAMES.get(exercise)
        self.X = None
        self.y = None
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load dataset dari CSV."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset {self.dataset_path} tidak ditemukan! Jalankan collect_data.py dulu.")
        
        df = pd.read_csv(self.dataset_path, header=None)
        print(f"📊 Loaded {len(df)} samples from {self.dataset_path}")
        
        # Pisahkan fitur dan label
        self.X = df.iloc[:, :-1].values  # Semua kolom kecuali terakhir
        self.y = df.iloc[:, -1].map(LABEL_MAP).values  # Label terakhir
        
        # Cek distribusi kelas
        unique, counts = np.unique(self.y, return_counts=True)
        print(f"📈 Class distribution: {dict(zip(['bad','good','perfect'], counts))}")
        
        return self.X, self.y
    
    def train_random_forest(self, test_size=0.2, random_state=42):
        """Train Random Forest classifier."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Grid search sederhana untuk hyperparameter
        best_rf = None
        best_acc = 0
        
        for n_estimators in [50, 100, 200]:
            for max_depth in [10, 20, None]:
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                acc = accuracy_score(y_test, rf.predict(X_test))
                
                if acc > best_acc:
                    best_acc = acc
                    best_rf = rf
        
        self.models['random_forest'] = best_rf
        print(f"✅ Random Forest accuracy: {best_acc:.2%}")
        return best_rf
    
    def train_svm(self, test_size=0.2, random_state=42):
        """Train SVM classifier."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        best_svm = None
        best_acc = 0
        
        for C in [0.1, 1, 10]:
            for gamma in ['scale', 'auto', 0.1]:
                svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=random_state)
                svm.fit(X_train, y_train)
                acc = accuracy_score(y_test, svm.predict(X_test))
                
                if acc > best_acc:
                    best_acc = acc
                    best_svm = svm
        
        self.models['svm'] = best_svm
        print(f"✅ SVM accuracy: {best_acc:.2%}")
        return best_svm
    
    def train_neural_network(self, test_size=0.2, random_state=42):
        """Train Neural Network classifier."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        best_nn = None
        best_acc = 0
        
        architectures = [
            (100, 50),      # 2 hidden layers
            (50, 25),       # Smaller
            (100, 50, 25),  # 3 hidden layers
        ]
        
        for hidden_layers in architectures:
            nn = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=random_state,
                early_stopping=True
            )
            nn.fit(X_train, y_train)
            acc = accuracy_score(y_test, nn.predict(X_test))
            
            if acc > best_acc:
                best_acc = acc
                best_nn = nn
        
        self.models['neural_network'] = best_nn
        print(f"✅ Neural Network accuracy: {best_acc:.2%}")
        return best_nn
    
    def ensemble_predict(self, X):
        """Ensemble prediction from all models."""
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        
        # Majority voting
        predictions = np.array(predictions)
        final_pred = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            final_pred.append(np.bincount(votes).argmax())
        
        return np.array(final_pred)
    
    def cross_validate_best_model(self, n_folds=5):
        """Cross-validation untuk best model."""
        if self.best_model is None:
            raise ValueError("Belum ada model yang dilatih!")
        
        cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=n_folds)
        
        print(f"\n📊 Cross-validation results ({n_folds}-folds):")
        print(f"   Mean accuracy: {cv_scores.mean():.2%}")
        print(f"   Std deviation: {cv_scores.std():.2%}")
        print(f"   Individual folds: {cv_scores}")
        
        return cv_scores
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluasi model secara komprehensif."""
        y_pred = model.predict(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['bad', 'good', 'perfect'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def save_model(self, model, name="best_model"):
        """Save model ke file."""
        filename = f"model_{self.exercise}.pkl"
        joblib.dump(model, filename)
        print(f"💾 Model saved to {filename}")
        return filename
    
    def train_all_and_select_best(self):
        """Train all models dan pilih yang terbaik."""
        print(f"\n{'='*50}")
        print(f"🏋️ Training models for {self.exercise.upper()}")
        print(f"{'='*50}\n")
        
        # Load data
        self.load_data()
        
        # Train all models
        self.train_random_forest()
        self.train_svm()
        self.train_neural_network()
        
        # Select best based on accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        best_acc = 0
        best_name = None
        
        for name, model in self.models.items():
            acc = accuracy_score(y_test, model.predict(X_test))
            if acc > best_acc:
                best_acc = acc
                best_name = name
                self.best_model = model
        
        print(f"\n🏆 Best model: {best_name} with accuracy {best_acc:.2%}")
        
        # Save best model
        self.save_model(self.best_model)
        
        # Return evaluation metrics
        return {
            'exercise': self.exercise,
            'best_model': best_name,
            'accuracy': best_acc,
            'models': {name: accuracy_score(y_test, model.predict(X_test)) 
                      for name, model in self.models.items()}
        }


def generate_training_report(results):
    """Generate laporan training untuk skripsi."""
    print("\n" + "="*60)
    print("📋 SKRIPSI: MODEL TRAINING REPORT")
    print("="*60)
    
    for ex, result in results.items():
        print(f"\n📌 {ex.upper()}")
        print(f"   Best Model: {result['best_model']}")
        print(f"   Accuracy: {result['accuracy']:.2%}")
        print(f"   Model Comparison:")
        for model_name, acc in result['models'].items():
            print(f"      - {model_name}: {acc:.2%}")
    
    print("\n" + "="*60)


def main():
    """Main function to train all models."""
    print("🚀 FitMove Model Training System")
    print("="*50)
    
    results = {}
    
    # Train untuk setiap olahraga yang memiliki dataset
    for exercise in ['curl', 'pushup', 'squat']:
        if os.path.exists(DATASETS[exercise]):
            trainer = ModelTrainer(exercise)
            result = trainer.train_all_and_select_best()
            results[exercise] = result
            
            # Cross-validation untuk best model
            trainer.cross_validate_best_model()
        else:
            print(f"⚠️ Dataset {DATASETS[exercise]} tidak ditemukan, skip {exercise}")
    
    # Generate report
    generate_training_report(results)
    
    print("\n✅ Training selesai! Model siap digunakan di app.py")


if __name__ == "__main__":
    main()