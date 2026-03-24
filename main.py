import numpy as np
import pickle

from src.backward import Backward
from src.forward import Forward
from src.hidden_layer import Hidden_layer
from src.loader import Loader
from src.saver import Saver
from src.trainer import Trainer
#from src.app import App

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = [
    {
        "age_years": 2.5,
        "mileage_km": 50000,
        "tread_depth_mm": 4.0,
        "repair_count": 1,
        "tire_type": 0,
        "pressure_kpa": 220,
        "usage_conditions": 0,
        "remaining_life_months": 12
    },
    {
        "age_years": 1.0,
        "mileage_km": 10000,
        "tread_depth_mm": 6.0,
        "repair_count": 0,
        "tire_type": 1,
        "pressure_kpa": 210,
        "usage_conditions": 1,
        "remaining_life_months": 24
    }
]

features = [
    "age_years", "mileage_km", "tread_depth_mm", "repair_count",
    "tire_type", "pressure_kpa", "usage_conditions"
]

output_data = np.array([[tire[feature] for feature in features] for tire in data])
input_data = np.array([[tire["remaining_life_months"]] for tire in data])

# Нормализация размеров данных для более быстрого обучения
scaler = StandardScaler()
output_data_scaled = scaler.fit_transform(output_data)

output_data_train, output_data_test, input_data_train, input_data_test = train_test_split(
    output_data_scaled, input_data, test_size = 0.2, random_state = 42
)

input_size = output_data_train.shape[1]
fan_in = np.sqrt(2 / input_size)

in_features = input_data.shape[1]
hidden_features = 8 #из-за маленького кол-ва данных, MSE при большом кол-ве нееронов будет больше
out_features = output_data.shape[1]

weights = [
    np.random.randn(in_features, hidden_features) * fan_in,
    np.random.randn(hidden_features, out_features) * fan_in
]

biases = [
    np.zeros(hidden_features),
    np.zeros(out_features)
]

print("output data shape:", output_data.shape)
print("input data shape:", input_data.shape)
print("W1 shape:", weights[0].shape)
print("W2 shape:", weights[1].shape)
print("b1 shape:", biases[0].shape)
print("b2 shape:", biases[1].shape)

# Обучение модели
trainer = Trainer(
    np,
    input_data_train, output_data_train,
    weights, biases,
    Backward, Forward,
    Hidden_layer, 0.01,
    1000
)

results = trainer.train()
for result in results:
    print(f'Epoch: {result['epoch']}, MSE: {result['mse']}')

saver = Saver(
    pickle, weights,
    biases, 'database.pkl'
)

saver.save()

#def main(): pass
#if __name__ == "__main__": main()
