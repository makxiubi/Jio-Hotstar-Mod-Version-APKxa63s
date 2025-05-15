# Step 1: Import library yang dibutuhkan
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset Iris
iris = load_iris()
X = iris.data  # Fitur (features)
y = iris.target  # Label (target)

# Step 3: Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Buat model K-Nearest Neighbors (KNN)
model = KNeighborsClassifier(n_neighbors=3)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Prediksi dan evaluasi model
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model: {accuracy * 100:.2f}%')

