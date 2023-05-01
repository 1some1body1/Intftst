# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, GRU, Dense

# Загружаем данные о курсе обмена Евро и Доллара
df = pd.read_csv('DAT_ASCII_EURUSD_M1_2022.csv', sep=' |;', engine='python', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], usecols=[2, 3, 4, 5])
# Отбираем только столбец Close и преобразуем его в массив numpy
data = df['Close'].values
data = data.reshape(-1, 1)

# Разделяем данные на обучающую и тестовую выборки (80% и 20% соответственно)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Нормализуем данные в диапазон от 0 до 1 с помощью MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Создаем функцию для преобразования данных в последовательности для обучения рекуррентных нейронных сетей
def create_sequences(data, window_size):
  X = []
  y = []
  for i in range(window_size, len(data)):
    X.append(data[i-window_size:i])
    y.append(data[i])
  return np.array(X), np.array(y)

# Задаем размер окна для обучения (30 дней)
window_size = 30

# Преобразуем обучающие и тестовые данные в последовательности
X_train, y_train = create_sequences(train_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

# Создаем модель рекуррентной нейронной сети с двумя слоями LSTM и одним слоем Dense
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(32))
model.add(Dense(1))

# Компилируем модель с функцией потерь MSE и оптимизатором Adam
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучаем модель на обучающих данных с размером батча 32 и количеством эпох 50
model.fit(X_train, y_train, batch_size=32, epochs=2)

# Предсказываем значения курса обмена на тестовых данных
y_pred = model.predict(X_test)

# Обратно преобразуем данные из нормализованного вида в исходный масштаб
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)

# Вычисляем среднеквадратичную ошибку предсказаний
mse = np.mean((y_test - y_pred)**2)
print(f'MSE: {mse:.4f}')

# Строим график фактических и предсказанных значений курса обмена
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('EUR/USD Exchange Rate Prediction with LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()