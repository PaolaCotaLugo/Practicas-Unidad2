
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Nueva importacion para las metricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

datos = yf.download('BTC-USD', start='2022-01-01', end='2024-01-01')
precios = datos[['Close']].values


escalador = MinMaxScaler(feature_range=(0, 1))
precios_esc = escalador.fit_transform(precios)

# Ventanas
X, y = [], []
for i in range(60, len(precios_esc)):
    X.append(precios_esc[i-60:i, 0])
    y.append(precios_esc[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

modelo = Sequential()
modelo.add(LSTM(units=50, input_shape=(X.shape[1], 1)))
modelo.add(Dense(units=1))
modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(X, y, epochs=5, batch_size=32)

# Pronostico
predicciones = modelo.predict(X)
predicciones = escalador.inverse_transform(predicciones)
reales = precios[60:]

rmse = np.sqrt(mean_squared_error(reales, predicciones))
mae = mean_absolute_error(reales, predicciones)
r2 = r2_score(reales, predicciones)

print(f"\nEvaluacion del Modelo ")
print(f"RMSE (Error Cuadratico Medio): {rmse:.2f}")
print(f"MAE (Error Absoluto Medio): {mae:.2f}")
print(f"R2 Score (Precision): {r2:.4f}")


# Grafica
plt.figure(figsize=(10, 6))
plt.plot(reales, color='blue', label='Precio Real')
plt.plot(predicciones, color='red', label='Prediccion LSTM')
plt.title('Pronostico de Bitcoin con Modelo LSTM')
plt.xlabel('Tiempo')
plt.ylabel('Precio USD')
plt.legend()
plt.show()



