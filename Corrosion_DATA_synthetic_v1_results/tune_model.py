import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    explained_variance_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, r2_score, root_mean_squared_error
)

base_url = "https://raw.githubusercontent.com/SushantGautam/amine_DATA_Modelling_synthetic_full/refs/heads/main/Corrosion_DATA_synthetic_v1/"
train_df = pd.read_csv(base_url + "TVAESynthesizer.csv")
test_df = pd.read_csv(base_url + "data.csv")
# predictor = TabularPredictor(label='Pit' ).fit(
#     train_data=train_df, ag_args_fit={'num_gpus': 1}, fit_strategy='parallel', presets=['best_quality'],
#     )
# preds = predictor.predict(test_df)
# print(preds)

########### predict ###########
model_path = "/home/sushant/D1/gcp_use/amine/AutogluonModels/ag-20250402_091114"
predictor = TabularPredictor.load(model_path)
preds = predictor.predict(test_df)
# If true labels exist in the test data
if 'Pit' in test_df.columns:
    scores = predictor.evaluate(test_df)
else:
    print("Warning: 'Pit' column not found in test_df. Evaluation metrics require ground truth.")
print(preds)

print("\nEvaluation Scores:")
scores = predictor.evaluate(test_df)
print(scores)
import matplotlib.pyplot as plt
y = test_df['Pit']
variable = 'Pit'

plt.figure(figsize=(8, 6))
plt.scatter(y, preds, alpha=0.6)
plt.xlabel('Original')
plt.ylabel('Predicted')
plt.title('Original vs. Predicted')
plt.grid(True)
plt.savefig('autogluon.png')
plt.show()
print("\nPredictor Summary:")
print(predictor)

print("\nModel Leaderboard:")
print(predictor.leaderboard(silent=True))

print("\nAll Models:")
print(predictor.model_names())

print("\nBest Model:")
print(predictor.model_best)
predictor.plot_ensemble_model()
y_metrics = [
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    r2_score
]
names = ["Explained Var", "MAE", "MSE", "RMSE", "Median AE", "R²"]
loss_dict = {}

print("\nCustom Metrics:")
for n, f in zip(names, y_metrics):
    loss = f(y, preds)
    loss_dict[n] = loss
    print(f"{n:>12}: {loss:.4f}")
# save loss_dict as custom_metrics.json
import json
import os
with open(os.path.join(model_path, "custom_metrics.json"), 'w') as f:
    json.dump(loss_dict, f)