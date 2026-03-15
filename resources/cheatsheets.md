# Deep Learning Cheatsheets and Quick References

## Neural Network Fundamentals

### Activation Functions

| Function | Formula | Range | Derivative | Use Case |
|----------|---------|-------|------------|----------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | (0, 1) | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ | Binary classification output |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | $\tanh'(x) = 1 - \tanh^2(x)$ | Hidden layers (zero-centered) |
| **ReLU** | $ReLU(x) = \max(0, x)$ | [0, ∞) | $ReLU'(x) = \begin{cases}1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Most common for hidden layers |
| **Leaky ReLU** | $LReLU(x) = \max(\alpha x, x)$ | (-∞, ∞) | $LReLU'(x) = \begin{cases}1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$ | Solves dying ReLU problem |
| **ELU** | $ELU(x) = \begin{cases}x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$ | (-α, ∞) | $ELU'(x) = \begin{cases}1 & x > 0 \\ \alpha e^x & x \leq 0 \end{cases}$ | Faster learning, reduces bias shift |
| **SELU** | $SELU(x) = \lambda \begin{cases}x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$ | (-λα, ∞) | Complex | Self-normalizing networks |
| **Softmax** | $\sigma_i(z) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0, 1) | Complex | Multi-class output |
| **Swish** | $Swish(x) = x \cdot \sigma(\beta x)$ | (-∞, ∞) | $Swish'(x) = \sigma(\beta x) + x \cdot \sigma(\beta x)(1 - \sigma(\beta x))\beta$ | Alternative to ReLU |

### Loss Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **MSE** | $L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Regression |
| **MAE** | $L = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$ | Regression (robust to outliers) |
| **Binary Cross-Entropy** | $L = -\frac{1}{n}\sum_{i=1}^n [y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ | Binary classification |
| **Categorical Cross-Entropy** | $L = -\sum_{i=1}^n y_i\log(\hat{y}_i)$ | Multi-class classification |
| **Hinge Loss** | $L = \max(0, 1 - y \cdot \hat{y})$ | SVM-style classification |
| **Focal Loss** | $L = -\alpha(1-\hat{y}_i)^\gamma \log(\hat{y}_i)$ | Imbalanced classification |
| **Triplet Loss** | $L = \max(0, d(a,p) - d(a,n) + margin)$ | Metric learning |
| **KL Divergence** | $L = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$ | Distribution matching |

### Optimization Algorithms

| Algorithm | Update Rule | Pros | Cons |
|-----------|-------------|------|------|
| **SGD** | $\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$ | Simple, low memory | Slow convergence, sensitive to learning rate |
| **Momentum** | $v_{t+1} = \beta v_t + \eta \nabla_\theta L(\theta_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ | Faster convergence, escapes local minima | Extra hyperparameter |
| **Nesterov** | $v_{t+1} = \beta v_t + \eta \nabla_\theta L(\theta_t + \beta v_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ | Better convergence than momentum | More complex |
| **AdaGrad** | $G_{t+1} = G_t + \nabla_\theta L(\theta_t)^2$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \nabla_\theta L(\theta_t)$ | Adapts learning rates per parameter | Learning rate decreases too fast |
| **RMSprop** | $v_{t+1} = \beta v_t + (1-\beta)\nabla_\theta L(\theta_t)^2$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \nabla_\theta L(\theta_t)$ | Solves AdaGrad decay problem | Still requires tuning |
| **Adam** | $m_{t+1} = \beta_1 m_t + (1-\beta_1)\nabla_\theta L(\theta_t)$<br>$v_{t+1} = \beta_2 v_t + (1-\beta_2)\nabla_\theta^2 L(\theta_t)$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \hat{m}_{t+1}$ | Fast, adaptive learning rates | May generalize poorly |
| **AdamW** | Adam with decoupled weight decay | Better generalization | Slightly more complex |
| **LAMB** | Layer-wise adaptive moments | Large batch training | Complex implementation |

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

#### ResNet (2015)
```
Input → Conv → [Residual Block]×N → Global Avg Pool → FC → Output
```

#### EfficientNet (2019)
```
Input → [MBConv Blocks with compound scaling] → FC → Output
```

### Advanced CNN Concepts

#### Dilated Convolutions
- **Output size**: $H_{out} = \lfloor\frac{H_{in} + 2P - K \times D}{S}\rfloor + 1$
- **Dilation rate**: $D$ controls spacing between kernel elements

#### Depthwise Separable Convolutions
- **Depthwise**: Apply one filter per input channel
- **Pointwise**: $1 \times 1$ convolution to combine channels
- **Reduces parameters**: $K \times K \times C_{in} \times C_{out} \rightarrow K \times K \times C_{in} + C_{in} \times C_{out}$

#### Attention Mechanisms
- **SE Block**: $y = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x))) \cdot x$
- **CBAM**: Channel + Spatial attention

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

### Positional Encoding
- **Sinusoidal**: $PE_{(pos,2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})$
- **Learned**: Embedding layer for positions

### Transformer Variants
- **BERT**: Bidirectional, masked language modeling
- **GPT**: Autoregressive, causal attention mask
- **T5**: Text-to-text, encoder-decoder
- **Vision Transformer**: Apply to image patches

## Training Techniques

### Regularization
- **L2 Regularization**: $L_{total} = L + \lambda \sum_i \theta_i^2$
- **L1 Regularization**: $L_{total} = L + \lambda \sum_i |\theta_i|$
- **Dropout**: Randomly set activations to zero during training
- **DropConnect**: Randomly set weights to zero
- **Batch Normalization**: Normalize layer inputs: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
- **Layer Normalization**: Normalize across features: $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- **Weight Normalization**: Reparameterize weights: $w = \frac{g}{\|v\|} v$

### Data Augmentation
- **Images**: Rotation, flip, crop, color jitter, mixup, cutmix
- **Text**: Back-translation, synonym replacement, random insertion/deletion
- **Audio**: Time stretching, pitch shifting, noise addition

### Learning Rate Scheduling
- **Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$
- **Exponential decay**: $\eta_t = \eta_0 \cdot e^{-\lambda t}$
- **Cosine annealing**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$
- **Warmup**: Gradually increase learning rate at start
- **Cyclical**: Oscillate between min and max learning rates

### Gradient Techniques
- **Gradient Clipping**: $\text{clip}(\nabla, \text{max\_norm}) = \frac{\nabla}{\max(1, \|\nabla\|/\text{max\_norm})}$
- **Gradient Accumulation**: Accumulate gradients over multiple batches
- **Mixed Precision**: Use FP16 for forward pass, FP32 for gradients

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
layers.Conv1D(filters, kernel_size, activation) # 1D convolution
layers.DepthwiseConv2D(kernel_size)            # Depthwise convolution
layers.SeparableConv2D(filters, kernel_size)    # Separable convolution
layers.MaxPooling2D(pool_size)                 # Max pooling
layers.AveragePooling2D(pool_size)             # Average pooling
layers.GlobalMaxPooling2D()                    # Global max pooling
layers.GlobalAveragePooling2D()                # Global avg pooling
layers.LSTM(units, return_sequences=True)      # LSTM
layers.GRU(units)                              # GRU
layers.Bidirectional(layer)                    # Bidirectional wrapper
layers.Attention()                             # Attention mechanism
layers.MultiHeadAttention(num_heads, key_dim)  # Multi-head attention
layers.BatchNormalization()                    # Batch norm
layers.LayerNormalization()                   # Layer norm
layers.Dropout(rate)                           # Dropout
layers.SpatialDropout2D(rate)                 # Spatial dropout
layers.Flatten()                               # Flatten
layers.Reshape(target_shape)                    # Reshape
layers.Permute(dims)                           # Permute dimensions
layers.RepeatVector(n)                         # Repeat vector
layers.Embedding(vocab_size, embed_dim)        # Embedding layer
```

### Callbacks
```python
keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
keras.callbacks.LearningRateScheduler(schedule_func)
keras.callbacks.TensorBoard(log_dir)
keras.callbacks.CSVLogger(filename)
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

### Common Layers
```python
nn.Linear(in_features, out_features)           # Fully connected
nn.Conv2d(in_channels, out_channels, kernel)   # 2D convolution
nn.Conv1d(in_channels, out_channels, kernel)   # 1D convolution
nn.MaxPool2d(kernel_size)                      # Max pooling
nn.AvgPool2d(kernel_size)                      # Average pooling
nn.LSTM(input_size, hidden_size, num_layers)    # LSTM
nn.GRU(input_size, hidden_size, num_layers)      # GRU
nn.BatchNorm2d(num_features)                    # Batch norm
nn.LayerNorm(normalized_shape)                  # Layer norm
nn.Dropout(p)                                   # Dropout
nn.Embedding(num_embeddings, embedding_dim)     # Embedding layer
nn.Flatten()                                    # Flatten
nn.Sequential(*layers)                          # Sequential container
```

### Activation Functions
```python
nn.ReLU()                                       # ReLU
nn.Sigmoid()                                    # Sigmoid
nn.Tanh()                                       # Tanh
nn.Softmax(dim=1)                               # Softmax
nn.LeakyReLU(negative_slope=0.01)              # Leaky ReLU
nn.ELU(alpha=1.0)                              # ELU
nn.GELU()                                       # GELU
```

## Evaluation Metrics

### Classification
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **Specificity**: $\frac{TN}{TN + FP}$
- **Sensitivity**: $\frac{TP}{TP + FN}$ (same as recall)

### Regression
- **MAE**: $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **MSE**: $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **RMSE**: $\sqrt{MSE}$
- **R²**: $1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$
- **MAPE**: $\frac{100\%}{n}\sum_{i=1}^n |\frac{y_i - \hat{y}_i}{y_i}|$

### Ranking & Recommendation
- **NDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Hit Rate**: Fraction of users with at least one relevant item
- **Recall@K**: Fraction of relevant items in top-K recommendations

### Custom Metrics
```python
def custom_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

model.compile(optimizer='adam', loss='mse', metrics=[custom_accuracy])
```

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
6. **NaN Loss**: Check for division by zero, log(0), or exploding gradients
7. **Imbalanced Data**: Use weighted loss, focal loss, or resampling
8. **Catastrophic Forgetting**: Use elastic weight consolidation or rehearsal

### Monitoring
- Track training and validation loss
- Use TensorBoard for visualization
- Monitor gradient norms
- Check for NaN values
- Verify data preprocessing
- Watch learning rate schedule
- Monitor weight distributions

### Performance Optimization
- **Mixed Precision Training**: Use FP16 for forward/backward, FP32 for updates
- **Gradient Checkpointing**: Trade compute for memory
- **Distributed Training**: Data parallel, model parallel
- **Model Pruning**: Remove unnecessary weights
- **Quantization**: Use lower precision weights
- **Knowledge Distillation**: Train smaller model with larger model guidance

## Advanced Topics

### Transfer Learning
- **Fine-tuning**: Unfreeze top layers of pre-trained model
- **Feature Extraction**: Freeze all layers, train only classifier
- **Domain Adaptation**: Adapt to different but related domains

### Generative Models
- **GAN**: Generator vs Discriminator adversarial training
- **VAE**: Variational Autoencoder with latent space
- **Diffusion Models**: Gradual noise addition/denoising
- **Flow Models**: Normalizing flows for density estimation

### Self-Supervised Learning
- **Contrastive Learning**: Learn representations by contrasting positives/negatives
- **Masked Language Modeling**: Predict masked tokens (BERT)
- **Rotation Prediction**: Predict image rotations
- **Jigsaw Puzzles**: Solve image puzzles

### Few-Shot Learning
- **Meta-Learning**: Learn to learn (MAML)
- **Prototypical Networks**: Learn class prototypes
- **Siamese Networks**: Learn similarity functions
- **Metric Learning**: Learn embedding space

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

### Tips and Tricks
- [Stanford CS230 Deep Learning Tips and Tricks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks/) - Practical advice for deep learning projects
