from model_builder.get_model import get_architected_cnn_model
from model_builder.utils import save_cnn
from data_builder.load_data import load_data
print(load_data)


X, y = load_data()

print(X.shape, y.shape)
# model = get_architected_cnn_model()
# compile_cnn(model)
# train_cnn(model, X, y)
# print('training is done')

