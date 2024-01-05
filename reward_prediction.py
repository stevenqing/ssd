import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow.keras import layers, models
import json
import numpy as np

saving_path = '/scratch/prj/inf_du/shuqing/reward_model'

loaded_model = tf.keras.models.load_model(saving_path)
# 1. 准备数据
obs = []
actions = []
rewards = []
obs_action = []
num_epochs = 100
batch_size = 32
count = 0

with open("/scratch/prj/inf_du/shuqing/trajs_file.json","r") as file:
    for line in file:
        try:
            data = json.loads(line)
            obs.append(tf.squeeze(np.array(data["vector_states"])))
            actions.append(tf.squeeze(np.array(data["actions"])))
            rewards.append(np.squeeze(np.array(data["rewards"])).tolist())
            i = tf.concat([tf.squeeze(np.array(data["vector_states"])), tf.squeeze(np.array(data["actions"]))], 0)
            i = i.numpy().tolist()
            obs_action.append(i)
            count+=1
            if count > 1000:
                break
        except:
            pass
print(np.shape(obs_action),np.shape(obs),np.shape(actions),np.shape(rewards))

# X1 = obs[:100]
# X2 = actions[:100]
# y = rewards[:100]

# X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(obs, actions, rewards, test_size=0.2, random_state=42)

# split the obs and action
X_train,X_test,y_train,y_test = train_test_split(obs_action,rewards, test_size=0.2, random_state=42)


# X1_train = tf.constant(X1_train)
# X2_train = tf.constant(X2_train)
# y_train = tf.constant(y_train)

X_train = tf.constant(X_train)
y_train = tf.constant(y_train)


# 数据标准化
'''
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_1':X1_train,'input_2':X2_train} , y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_1':X1_test,'input_2':X2_test} , y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train , y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test , y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)
'''

'''

# 2. 构建模型
# input_placeholder_1 = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,39,32,3))
# input_placeholder_2 = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,5))
'''


'''
# 定义第一个输入
input_1 = tf.keras.layers.Input(shape=(39,32,3), name='input_1') # (input_placeholder_1)
conv_1 = layers.Conv2D(16, (3, 3), activation='relu')(input_1)
pool1 = layers.MaxPooling2D((2, 2))(conv_1)
flat1 = layers.Flatten()(pool1)
dense1_1 = layers.Dense(1024,activation='relu')(flat1)
dense1_2 = layers.Dense(256,activation='relu')(flat1)(dense1_1)

# 定义第二个输入
input_2 = tf.keras.layers.Input(shape=(5), name='input_2') # (input_placeholder_2)
flat2 = layers.Flatten()(input_2)
dense2_1 = layers.Dense(16, activation='relu')(flat2)
dense2_2 = layers.Dense(16, activation='relu')(dense2_1)

# 将两个输入合并
merged = layers.concatenate([dense1_2, dense2_2])

# 添加全连接层
dense1 = layers.Dense(128, activation='relu')(merged)
dense2 = layers.Dense(64, activation='relu')(dense1)

# 输出层
output = layers.Dense(5, activation='sigmoid')(dense2)

'''

#print(X_train[1])
model = models.Sequential([
    layers.Flatten(input_shape=(np.shape(X_train)[1],)),  # 将28x28的图像展平成一维数组
    layers.Dense(1024, activation='relu'),   # 全连接层，128个神经元，使用ReLU激活函数
    layers.Dense(512, activation='relu'),
	layers.Dense(128, activation='relu'),
	layers.Dense(np.shape(y_train)[1], activation='sigmoid')   # 输出层，使用Sigmoid激活函数进行二元分类
])



# 创建模型
# model = models.Model(inputs=[input_1, input_2], outputs=output)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,clipvalue=2.0)
# 编译模型
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'])

# 4. 训练模型

# 定义损失函数和优化器

# 准备训练数据
'''
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 前向传播
            logits = model(x_batch)
            loss_value = loss_fn(y_batch, logits)
            print(loss_value) 
        # 计算梯度
        gradients = tape.gradient(loss_value, model.trainable_variables)
        
        # 打印梯度信息
        #for var, grad in zip(model.trainable_variables, gradients):
            #print(f"{var.name}: {grad.numpy().mean()}")
        
        # 更新模型参数
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size, validation_data=test_dataset)
'''


model.fit(X_train,y_train,epochs=1,batch_size=batch_size)

#model.save(saving_path)
tf.saved_model.save(model,saving_path)
# 5. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# 6. 进行预测
new_data = scaler.transform(new_data)  # 标准化新数据
predictions = model.predict(new_data)

