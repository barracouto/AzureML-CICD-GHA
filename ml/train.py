# Simple Iris training
import os, sys, joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# default
out_dir = "outputs"

# accept either --output or --output_dir
argv = sys.argv
if "--output" in argv:
    out_dir = argv[argv.index("--output") + 1]
elif "--output_dir" in argv:
    out_dir = argv[argv.index("--output_dir") + 1]

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Accuracy={acc:.3f}")

os.makedirs(out_dir, exist_ok=True)
joblib.dump(clf, os.path.join(out_dir, "model.joblib"))
print(f"Saved to {os.path.join(out_dir, 'model.joblib')}")
