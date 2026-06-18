
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


class IrisClassifier:
    

    def __init__(self, csv_path, n_neighbors=5):
        
        self.csv_path = csv_path
        self.n_neighbors = n_neighbors

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.predictions = None

    def load_data(self):
        
        self.df = pd.read_csv(self.csv_path)
        print("Data loaded successfully.")
        print(f"Shape: {self.df.shape}")
        return self.df

    def explore_data(self):
        
        print("\n--- First 5 rows ---")
        print(self.df.head())

        print("\n--- Statistical Summary ---")
        print(self.df.describe())

        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())

        print("\n--- Species Count ---")
        print(self.df['species'].value_counts())

    def prepare_features(self):
       
        self.X = self.df.drop('species', axis=1).values
        self.y = self.df['species'].values
        print("\nFeatures and target separated.")
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

    def split_data(self, test_size=0.2, random_state=42):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        print(f"\nTraining samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")

    def scale_features(self):
       
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("\nFeatures scaled (mean=0, variance=1).")

    def find_best_k(self, k_range=range(1, 21)):
       
        error_rates = []

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            preds = knn.predict(self.X_test)
            error = np.mean(preds != self.y_test)
            error_rates.append(error)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), error_rates, marker='o', linestyle='--', color='blue')
        plt.xlabel('K Value')
        plt.ylabel('Error Rate')
        plt.title('Elbow Method - Choosing Optimal K')
        plt.savefig('elbow_curve.png')
        plt.show()

        best_k = list(k_range)[np.argmin(error_rates)]
        print(f"\nBest K found: {best_k}")
        return best_k

    def train_model(self):
       
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        print(f"\nModel trained with K={self.n_neighbors}.")

    def predict(self):
        
        self.predictions = self.model.predict(self.X_test)
        print("\nPredictions generated on test data.")
        return self.predictions

    def evaluate(self):
        
        cm = confusion_matrix(self.y_test, self.predictions)
        species_labels = sorted(self.df['species'].unique())

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=species_labels,
            yticklabels=species_labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

        f1 = f1_score(self.y_test, self.predictions, average='weighted')
        print(f"\nF1 Score: {f1:.4f}")

        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, self.predictions))

    def run(self):
       
        self.load_data()
        self.explore_data()
        self.prepare_features()
        self.split_data()
        self.scale_features()
        self.train_model()
        self.predict()
        self.evaluate()


if __name__ == "__main__":
    classifier = IrisClassifier(csv_path="iris_dataset.csv", n_neighbors=5)
    classifier.run()
