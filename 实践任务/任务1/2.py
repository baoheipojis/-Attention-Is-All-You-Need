# 这段代码只能说能跑，实际上MSE一直是20多，说明线性模型已经无法处理该问题了。
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 数据准备
# 生成随机表达式和结果
def generate_data(num_samples):
    expressions = []
    results = []
    for _ in range(num_samples):
        expression = ""
        result = 0
        for _ in range(np.random.randint(1, 5)):  # 生成1到4个数字的加减表达式
            num = np.random.randint(1, 10)
            operation = np.random.choice(['+', '-'])
            expression += str(num) + operation
        result = eval(expression[:-1])  # 计算表达式的结果
        expressions.append(expression)
        results.append(result)
    return expressions, results

# 生成1000个样本
expressions, results = generate_data(1000)

# 将结果转换为numpy数组
results = np.array(results)

# 2. 模型设计
# 文本向量化
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(expressions)
sequences = tokenizer.texts_to_sequences(expressions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# 神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 3. 训练模型
model.fit(padded_sequences, results, epochs=100, verbose=1)

# 返回训练好的模型和tokenizer
model, tokenizer, max_length
# 假设这是您的新表达式
new_expressions = ["2+3", "4-1", "5+2-3","6+4-2"]

# 使用相同的tokenizer将新表达式转换为序列
new_sequences = tokenizer.texts_to_sequences(new_expressions)

# 填充序列以确保它们具有相同的长度
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')

# 使用模型进行预测
predictions = model.predict(new_padded_sequences)

# 打印预测结果
for expr, pred in zip(new_expressions, predictions):
    print(f"Expression: {expr}, Predicted Result: {pred[0]:.2f}")
