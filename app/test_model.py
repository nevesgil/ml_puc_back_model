import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from utils.data_methods import DataReader, NullRemover, DataTransformer

# Load the model
model_path = os.path.join(
    os.path.dirname(__file__), "../ml_model/spaceship_pipeline.pkl"
)
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Test data
reader = DataReader(
    "https://raw.githubusercontent.com/nevesgil/ml_puc_back_model/main/data/test.csv"
)
df = reader.read_csv()

null_remover = NullRemover(df)
df = null_remover.remove_nulls()

transformer = DataTransformer(df)
df = transformer.sum_spending_columns()
df = transformer.map_categorical_columns()
df = transformer.drop_columns()
df = transformer.convert_bool_columns()

test_data = df.drop("Transported", axis=1).values
true_labels = df["Transported"].values

# Thresholds for acceptable model performance
ACCURACY_THRESHOLD = 0.70
PRECISION_THRESHOLD = 0.70
RECALL_THRESHOLD = 0.60
F1_THRESHOLD = 0.70


# Test
@pytest.mark.parametrize("input_data, true_labels", [(test_data, true_labels)])
def test_model_performance(input_data, true_labels):
    predictions = model.predict(input_data)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy {accuracy} is below threshold!"
    assert (
        precision >= PRECISION_THRESHOLD
    ), f"Precision {precision} is below threshold!"
    assert recall >= RECALL_THRESHOLD, f"Recall {recall} is below threshold!"
    assert f1 >= F1_THRESHOLD, f"F1-score {f1} is below threshold!"
