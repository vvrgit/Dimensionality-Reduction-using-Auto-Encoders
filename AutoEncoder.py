import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np



class AutoEncoder:
    def __init__(self, in_features:int, latent_space:int, X_train, X_test, batch_size:int=64, save_model:bool=True) -> None:
        
        self.X_train = X_train
        self.X_test = X_test
        self.save_model = save_model
        self.latent_space = latent_space
        self.in_features = in_features
        self.batch_size = batch_size
        self.auto_encoder = None
        self._history = None
        self.performance_metrics = {}
        self.encoder = None
        self.compressed_data_train = None
        self.compressed_data_test = None
    
    def build_model(self):
        layers = ((self.in_features - self.latent_space)*2)+1 # no of stacked layers in the network
        latent_space_position = layers // 2
        model_layers = [0]*layers

        k = 1

        """
        for making the code dynamic. we can play with the input features, latent space size without modifying code.
        """
        for i in range(layers):
            if i == 0:
                model_layers[i] = Input(shape=(self.in_features), name='encoder_input')
            if i == latent_space_position:
                model_layers[i] = Dense(self.latent_space, activation='relu', name='latent_space')(model_layers[i-1])
            # encoder part
            if i > 0 and i < latent_space_position:
                model_layers[i] = Dense(self.in_features-i, activation='relu', name=f'encoder_layer{i}')(model_layers[i-1])

            # decoder part
            if i > latent_space_position and i < layers-1:
                model_layers[i] = Dense(self.latent_space+k, activation='relu', name=f'decoder_layer{i-latent_space_position}')(model_layers[i-1])
                k += 1
            if i == layers-1:
                model_layers[i] = Dense(self.in_features, activation='sigmoid', name='Output')(model_layers[i-1])
        
        modelname = f'AutoEncoder_LS_{self.latent_space}_BS_{self.batch_size}'
        self.auto_encoder = Model(model_layers[0], model_layers[-1], name=modelname)
        self.auto_encoder.summary()
        return self.auto_encoder
            
    def fit(self, epochs:int=50, save_model:bool=True, return_compressed_data:bool=False):
        self.auto_encoder.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

        print(f"Batch Size: {self.batch_size}")
        print(f'Epochs: {epochs}')

        # for auto encoders both x and y will be same
        self._history = self.auto_encoder.fit(self.X_train, # x
                                             self.X_train, # y
                                             epochs=epochs,
                                             steps_per_epoch=10,
                                             validation_data=(self.X_test,self.X_test),
                                             batch_size=self.batch_size)
        
        self.performance_metrics['Train Loss'] = self._history.history['loss'][-1]
        self.performance_metrics['Validation Loss'] = self._history.history['val_loss'][-1]
        
        print("\n-----------------------------------------------------------------------")
        print(f"\nTraining Loss: {self.performance_metrics['Train Loss']}")
        print(f"Training Loss: {self.performance_metrics['Validation Loss']}\n")
        print("-----------------------------------------------------------------------\n")
        
        latentSpace_layer = len(self.auto_encoder.layers)//2 #latent space will be at middle of the network
        enc_input = self.auto_encoder.inputs[0]
        enc_output = self.auto_encoder.layers[latentSpace_layer].output
        
        # only the encoder part
        encoder = Model(inputs=enc_input, outputs=enc_output)
        encoder.compile(optimizer='adam',
                        loss=tf.keras.losses.mean_squared_error)
        
        compressed_data_train = encoder.predict(self.X_train)
        compressed_data_test = encoder.predict(self.X_test)
        
        self.compressed_data_train = pd.DataFrame(compressed_data_train, columns=[f"X{x}" for x in range(1, compressed_data_train.shape[1]+1)])
        self.compressed_data_test = pd.DataFrame(compressed_data_test, columns=[f"X{x}" for x in range(1, compressed_data_test.shape[1]+1)])
        
        self.encoder = encoder
        
        if save_model:
            os.makedirs(f'exported_data/latent_space_{self.latent_space}_batch_{self.batch_size}', exist_ok=True)
            encoder.save(f'exported_data/latent_space_{self.latent_space}_batch_{self.batch_size}/encoder_ls{self.latent_space}_bs_{self.batch_size}.h5')
            self.compressed_data_train.to_csv(f'exported_data/latent_space_{self.latent_space}_batch_{self.batch_size}/compressed_data_train_ls{self.latent_space}_bs_{self.batch_size}.csv')
            self.compressed_data_train.to_csv(f'exported_data/latent_space_{self.latent_space}_batch_{self.batch_size}/compressed_data_test_ls{self.latent_space}_bs_{self.batch_size}.csv')
        
        print('\n\nLow-Dimensional Representation of Train Data:')
        display(self.compressed_data_train)
        
        if return_compressed_data:
            return self.compressed_data_train, self.compressed_data_test
    
    def metrics(self):
        return self.performance_metrics
    
    def visualize(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        axes[0].plot(self._history.history['loss'])
        axes[0].plot(self._history.history['val_loss'])
        axes[0].legend(labels=[f"Train Loss : {self.performance_metrics['Train Loss']:.7f}",
                        f"Test loss : {self.performance_metrics['Validation Loss']:.7f}"])
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f"Training Loss and Validation Loss Vs Epochs\nLatent Space {self.latent_space} and batch size {self.batch_size}")
        
        i = np.random.randint(1, len(self.X_test))
        out = self.auto_encoder.predict([self.X_test])
        axes[1].plot(self.X_test.iloc[i], lw=1.5, color='black')
        axes[1].plot(out[i], lw=1.5, color='blue')
        axes[1].fill_between(np.arange(12), self.X_test.iloc[i], out[i], color='lightcoral')
        axes[1].legend(['Original Data', 'Reconstructed Data'])
        axes[1].set_title('Original Vs Reconstructed Data')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Values')
