# Simple Iris training
import os, joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

out_dir = "outputs"
if "--output" in sys.argv:
    out_dir = sys.argv[sys.argv.index("--output") + 1]

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Accuracy={acc:.3f}")

os.makedirs("outputs", exist_ok=True)
joblib.dump(clf, "outputs/model.joblib")
print("Saved to outputs/model.joblib")
