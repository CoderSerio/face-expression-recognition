from utils import load_data, train, predict, draw_image_prediction

x_train, x_test, y_train, y_test = load_data('dataset/images/train')
model = train(x_train, y_train, 'model/face_expression_recognizer.keras')
predictions = predict(model, x_test, y_test)
# draw_image_prediction(x_test, predictions)
