import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import time
import csv

# Load dataset
dataSet = 'datasets/Nazario.csv'
dataFrame = pandas.read_csv(dataSet)
print("\nRunning", dataSet, " dataset")

# Handle missing values
dataFrame['body'] = dataFrame['body'].fillna('')

# Split into features (X) and targets (y)
features = dataFrame['body']
target = dataFrame['label']

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

# Split into training and testing sets 80% train, 20% test
featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(
    features, target, test_size=0.2, random_state=228
)

# Initialise the model
aiModel = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=228)

# Start time - for measuring
start_time = time.time()

# start training the model
aiModel.fit(featuresTrain, targetTrain)

# predict on the testing data
targetPredicted = aiModel.predict(featuresTest)

# Evaluate the performance
end_time = time.time()
elapsed_time = end_time - start_time
accuracy = accuracy_score(targetTest, targetPredicted)
report = classification_report(targetTest, targetPredicted, output_dict=True)
results = [
    ['Dataset', 'Class (0=Legit, 1=Spam)', 'Overall Accuracy', 'Precision', 'Recall', 'F1-score', 'Support', 'Time Taken (seconds)']
]

# Add rows for each class
for label, metrics in report.items():
    if isinstance(metrics, dict):
        results.append([
            dataSet,
            label,
            f"{accuracy:.4f}",
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            int(metrics['support']),
            f"{elapsed_time:.2f}"
        ])

# Write results to CSV
with open('Results_Nazario.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)

print("\n Results saved to 'Results_Nazario.csv'")
