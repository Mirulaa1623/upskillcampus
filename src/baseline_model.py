import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_save(X, y, model_path):
    # Flatten sequences
    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=15,
        random_state=42,
        n_jobs=2
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    joblib.dump(model, model_path)

    return rmse