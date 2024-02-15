from utils import load_data, train

x_train, x_test, y_train, y_test = load_data('dataset/images/train')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = train(x_train, y_train, 'model/face_expression_recognizer.keras'
              )
