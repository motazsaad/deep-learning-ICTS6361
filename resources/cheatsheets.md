# Deep Learning Cheatsheets and Quick References

## Neural Network Fundamentals

### Activation Functions

| Function | Formula | Range | Derivative | Use Case |
|----------|---------|-------|------------|----------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | (0, 1) | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ | Binary classification output |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | $\tanh'(x) = 1 - \tanh^2(x)$ | Hidden layers (zero-centered) |
| **ReLU** | $ReLU(x) = \max(0, x)$ | [0, ∞) | $ReLU'(x) = \begin{cases}1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Most common for hidden layers |
| **Leaky ReLU** | $LReLU(x) = \max(\alpha x, x)$ | (-∞, ∞) | $LReLU'(x) = \begin{cases}1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$ | Solves dying ReLU problem |
| **Softmax** | $\sigma_i(z) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0, 1) | Complex | Multi-class output |

### Loss Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **MSE** | $L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Regression |
| **Binary Cross-Entropy** | $L = -\frac{1}{n}\sum_{i=1}^n [y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ | Binary classification |
| **Categorical Cross-Entropy** | $L = -\sum_{i=1}^n y_i\log(\hat{y}_i)$ | Multi-class classification |
| **Hinge Loss** | $L = \max(0, 1 - y \cdot \hat{y})$ | SVM-style classification |

### Optimization Algorithms

| Algorithm | Update Rule | Pros | Cons |
|-----------|-------------|------|------|
| **SGD** | $\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$ | Simple, low memory | Slow convergence, sensitive to learning rate |
| **Momentum** | $v_{t+1} = \beta v_t + \eta \nabla_\theta L(\theta_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ | Faster convergence, escapes local minima | Extra hyperparameter |
| **Adam** | $m_{t+1} = \beta_1 m_t + (1-\beta_1)\nabla_\theta L(\theta_t)$<br>$v_{t+1} = \beta_2 v_t + (1-\beta_2)\nabla_\theta^2 L(\theta_t)$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \hat{m}_{t+1}$ | Fast, adaptive learning rates | May generalize poorly |

## Convolutional Neural Networks

### Convolution Operations

#### 2D Convolution
- **Output size**: $H_{out} = \lfloor\frac{H_{in} + 2P - K}{S}\rfloor + 1$
- **Same padding**: $P = \lfloor\frac{K-1}{2}\rfloor$
- **Valid padding**: $P = 0$

#### Pooling Operations
- **Max Pooling**: $\text{output} = \max_{i,j \in \text{window}} \text{input}_{i,j}$
- **Average Pooling**: $\text{output} = \frac{1}{|\text{window}|}\sum_{i,j \in \text{window}} \text{input}_{i,j}$

### Common CNN Architectures

#### LeNet-5 (1998)
```
Input (32x32) → Conv (6@28x28) → Pool (6@14x14) → Conv (16@10x10) → Pool (16@5x5) → FC (120) → FC (84) → Output (10)
```

#### AlexNet (2012)
```
Input (227x227x3) → Conv (96@55x55) → Pool → Conv (256@27x27) → Pool → Conv (384@13x13) → Conv (384@13x13) → Conv (256@13x13) → Pool → FC (4096) → FC (4096) → Output (1000)
```

#### VGG-16 (2014)
```
Input (224x224x3) → [Conv×3 + Pool]×2 → [Conv×3 + Pool]×3 → FC×3 → Output
```

## Recurrent Neural Networks

### RNN Equations
- **Hidden state**: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
- **Output**: $y_t = W_{hy}h_t + b_y$

### LSTM Equations
- **Forget gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **Input gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- **Candidate**: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- **Cell state**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- **Output gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- **Hidden state**: $h_t = o_t \odot \tanh(C_t)$

### GRU Equations
- **Update gate**: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$
- **Reset gate**: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
- **New memory**: $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$
- **Final memory**: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

## Transformers

### Self-Attention
- **Query**: $Q = XW_Q$
- **Key**: $K = XW_K$
- **Value**: $V = XW_V$
- **Attention**: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

### Multi-Head Attention
- Split Q, K, V into h heads
- Apply attention to each head
- Concatenate and project: $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

## Training Techniques

### Regularization
- **L2 Regularization**: $L_{total} = L + \lambda \sum_i \theta_i^2$
- **Dropout**: Randomly set activations to zero during training
- **Batch Normalization**: Normalize layer inputs: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

### Learning Rate Scheduling
- **Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$
- **Exponential decay**: $\eta_t = \eta_0 \cdot e^{-\lambda t}$
- **Cosine annealing**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$

## TensorFlow/Keras Quick Reference

### Model Creation
```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential API
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Common Layers
```python
layers.Dense(units, activation=None)           # Fully connected
layers.Conv2D(filters, kernel_size, activation) # 2D convolution
layers.MaxPooling2D(pool_size)                 # Max pooling
layers.LSTM(units, return_sequences=True)      # LSTM
layers.GRU(units)                              # GRU
layers.BatchNormalization()                    # Batch norm
layers.Dropout(rate)                           # Dropout
layers.Flatten()                               # Flatten
layers.Reshape(target_shape)                    # Reshape
```

### Training
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

test_loss, test_acc = model.evaluate(X_test, y_test)
```

## PyTorch Quick Reference

### Model Creation
```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

### Training Loop
```python
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Evaluation Metrics

### Classification
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
- **AUC-ROC**: Area under ROC curve

### Regression
- **MAE**: $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **MSE**: $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **RMSE**: $\sqrt{MSE}$
- **R²**: $1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

## Common Hyperparameters

| Hyperparameter | Typical Range | Effect |
|----------------|---------------|--------|
| Learning Rate | 1e-4 to 1e-2 | Step size for optimization |
| Batch Size | 16 to 512 | Memory usage, gradient noise |
| Number of Layers | 2 to 100+ | Model capacity |
| Hidden Units | 32 to 1024 | Representation power |
| Dropout Rate | 0.1 to 0.5 | Regularization strength |
| Weight Decay | 1e-6 to 1e-3 | L2 regularization |

## Debugging Tips

### Common Issues
1. **Vanishing/Exploding Gradients**: Use proper initialization, batch norm, or gradient clipping
2. **Overfitting**: Add regularization, dropout, or more data
3. **Underfitting**: Increase model capacity or train longer
4. **Poor Convergence**: Adjust learning rate, use better optimizer
5. **Memory Issues**: Reduce batch size, use gradient checkpointing

### Monitoring
- Track training and validation loss
- Use TensorBoard for visualization
- Monitor gradient norms
- Check for NaN values
- Verify data preprocessing

## Resources

### Official Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Keras Documentation](https://keras.io/api/)

### Tutorials and Courses
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai](https://course.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [MIT Introduction to Deep Learning](http://introtodeeplearning.com/)

### Papers and Surveys
- [Papers with Code](https://paperswithcode.com/)
- [arXiv CS.LG](https://arxiv.org/list/cs.LG/recent)
- [Awesome Deep Learning](https://github.com/owainlewis/awesome-deep-learning)
