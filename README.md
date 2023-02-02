# Dimensionality Reduction using Auto Encoders

Auto-encoders are neural network models used for unsupervised learning. They are trained to reconstruct the input data by encoding it into a lower-dimensional representation (latent space) and then decoding it back to its original form. During the training process, the auto-encoder learns to identify the most important features in the input data and compress them into the latent space, effectively reducing the dimensionality of the data. This learned compression can then be used for dimensionality reduction for new, unseen data, making it a useful tool for data visualization, feature extraction, and preprocessing for other machine learning tasks.

In this work, the dataset with 12 input features is compressed into 7 features and the reconstructed data is almost similar to the original input data.

[plots](https://github.com/PrabhuKiran8790/Dimensionality-Reduction-using-Auto-Encoders/blob/main/plots.png)

as we can see, the red shaded region tells us the error between original and reconstructed data
