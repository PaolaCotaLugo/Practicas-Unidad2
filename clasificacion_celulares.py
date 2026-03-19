

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Cargar datos
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv"
datos = pd.read_csv(url)

print(datos.describe())

X = datos.drop('price_range', axis=1)
y = datos['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

acc_rf = accuracy_score(y_test, rf.predict(X_test))
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))

# Graficar
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt='d', cmap='Greens', ax=ax[0])
ax[0].set_title('Matriz de Confusion - Random Forest')
ax[0].set_xlabel('Prediccion')
ax[0].set_ylabel('Valor Real')

sns.heatmap(confusion_matrix(y_test, mlp.predict(X_test)), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('Matriz de Confusion - MLP')
ax[1].set_xlabel('Prediccion')
ax[1].set_ylabel('Valor Real')

plt.tight_layout()
plt.show()

modelos = ['Random Forest', 'Red Neuronal (MLP)']
puntajes = [acc_rf, acc_mlp]

plt.figure(figsize=(8, 6))
plt.bar(modelos, puntajes, color=['green', 'blue'])
plt.ylim(0, 1)
plt.ylabel('Precision (Accuracy)')
plt.title('Comparacion de Modelos: Cual es mejor?')

# Mostrar valores sobre las barras
for i, v in enumerate(puntajes):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()

