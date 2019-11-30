from tensorflow.keras.models import load_model

def compile_cnn(model, optimizer='adam', loss='mean_squared_error', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train_cnn(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)

def save_cnn(model, fileName):
    model.save(fileName + '.h5')

def load_cnn(fileName):
    return load_model(fileName + '.h5')