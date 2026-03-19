import numpy as np
import pickle

from src.model import Model
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

# X - данные, Y - факты
X = np.array([[tire[feature] for feature in features] for tire in data])
y = np.array([[tire["remaining_life_months"]] for tire in data])

# Нормализация размеров данных для более быстрого обучения
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Обучение модели
model = Model(np, pickle, input_dim=X_train.shape[1], lr=0.01)
model.train(X_train, y_train, epochs=1000)

# Предсказание и оценка
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Среднеквадратичная ошибка на тесте: {mse:.3f}")

# Сохранение весов
model.save_weights("tire_model_weights.pkl")

#def main(): pass
#if __name__ == "__main__": main()
