import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow.keras import layers, models
import json
import numpy as np
# 1. 准备数据
obs = []
actions = []
rewards = []
num_epochs = 100
batch_size = 32
count = 0
with open("/scratch/prj/inf_du/shuqing/trajs_file.json","r") as file:
    for line in file:
        try:
            if count <= 1000:
                data = json.loads(line)
                obs.append(tf.squeeze(np.array(data["obs"])))
                actions.append(tf.squeeze(np.array(data["actions"])))
                rewards.append(tf.squeeze(np.array(data["rewards"])))
                count+=1
            else:
                break
        except:
            pass
print(np.shape(obs),np.shape(actions),np.shape(rewards))


X1 = obs[:100]
X2 = actions[:100]
y = rewards[:100]

# X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# X1_train = tf.constant(X1_train)
# X2_train = tf.constant(X2_train)
# y_train = tf.constant(y_train)




# 数据标准化
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_1':X1,'input_2':X2} , y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
# test_dataset = tf.data.Dataset.from_tensor_slices(({'input_1':X1_test,'input_2':X2_test} , y_test))

# 2. 构建模型

# 定义第一个输入
input_1 = tf.keras.layers.Input(shape=(39,32,3), name='input_1')
conv_1 = layers.Conv2D(16, (3, 3), activation='relu')(input_1)
pool1 = layers.MaxPooling2D((2, 2))(conv_1)
flat1 = layers.Flatten()(pool1)
dense1 = layers.Dense(1024,activation='relu')(flat1)
dense2 = layers.Dense(256,activation='relu')(dense1)
# 定义第二个输入
input_2 = tf.keras.layers.Input(shape=(5), name='input_2')
flat2 = layers.Flatten()(input_2)
dense2_1 = layers.Dense(16, activation='relu')(flat2)
dense2_2 = layers.Dense(16, activation='relu')(dense2_1)

print(dense1)
print(dense2_1)


# 将两个输入合并
merged = layers.concatenate([dense2, dense2_2])

# 添加全连接层
dense1 = layers.Dense(128, activation='relu')(merged)
dense2 = layers.Dense(64, activation='relu')(dense1)

# 输出层
output = layers.Dense(5, activation='softmax')(dense2)

# 创建模型
model = models.Model(inputs=[input_1, input_2], outputs=output)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)

# 5. 评估模型
test_loss, test_acc = model.evaluate([X1_test,X2_test], y_test)
print(f"Test Accuracy: {test_acc}")

# 6. 进行预测
new_data = scaler.transform(new_data)  # 标准化新数据
predictions = model.predict(new_data)

