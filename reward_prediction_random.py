import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
# 1. 准备数据
# X, y = make_classification(n_samples=50000, n_features=3794, n_classes=2, random_state=42)

X = np.random.randint(0,255,size=(50000,3794))
y = np.random.randint(0,1,size=(50000,5))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 构建模型
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(512, activation='relu'), 
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(5, activation='sigmoid')
])

# 3. 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# 5. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# 6. 进行预测
new_data = scaler.transform(new_data)  # 标准化新数据
predictions = model.predict(new_data)

