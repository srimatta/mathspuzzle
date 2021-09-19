import numpy as np
from PIL import Image
import pytesseract
from cv2 import cv2
import tensorflow as tf


def get_features_labes_from_image(image_file, features_lables_splitter ='=',  features_splitter = ','):
    """
    convert images files to text
    """
    img = Image.open(image_file)  # the second one

    text = pytesseract.image_to_string(img)

    lines = text.split('\n')
    features = []
    labels = []
    predict_features = []

    for line in lines:
        features_labels = line.split(features_lables_splitter)

        if(len(features_labels) == 2):
            features_ = features_labels[0].split(features_splitter)
            labels_ = features_labels[1].split(features_splitter)
            if('?' not in line):
                features_ = list(map(lambda x: int(x), features_))
                labels_ = list(map(lambda x: int(x), labels_))
                features.append(features_)
                labels.append(labels_)
            else:
                predict_features.append(list(map(lambda x: int(x), features_)))

    print(features)
    print(labels)
    print(predict_features)

    return (np.asarray(features), np.asarray(labels) , np.asarray(predict_features))


def build_math_model(X_train, Y_train, X_valid, Y_valid, epochs = 36000, batch_size = 1, verbose = 0):

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000),
    ]

    import numpy as np
    from keras.models import Sequential

    # Import `Dense` from `keras.layers`
    from keras.layers import Dense

    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(1, activation='relu', input_shape=(3,)))
    # Add an output layer
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs = epochs, batch_size = batch_size, verbose = verbose)

    print(model.summary())

    model.save('../model/model.h5')

    return model

def print_weights(model):
    # Dump weights
    for layerNum, layer in enumerate(model.layers):
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]

        for toNeuronNum, bias in enumerate(biases):
            print(f'{layerNum}B -> L{layerNum + 1}N{toNeuronNum}: {bias}')

        for fromNeuronNum, wgt in enumerate(weights):
            for toNeuronNum, wgt2 in enumerate(wgt):
                print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum + 1}N{toNeuronNum} = {wgt2}')


def get_trained_model():
    from keras.models import load_model

    # load model
    model = load_model('../model/model.h5')
    # summarize model.
    model.summary()

    return model


def draw_image_with_predicted(input_image_file, output_image_file ='../data/maths_puzzle_solved.png', target_label='?', predicted_value ="0"):

    # read the image and get the dimensions
    img = cv2.imread(input_image_file)
    h, w, _ = img.shape  # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img)  # also include any config options you use

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        if(b[0].strip() == '?'):
            print("target_label :", target_label)
            cv2.putText(img, predicted_value, (int(b[1]), h - int(b[2])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    cv2.imwrite(output_image_file, img)

if __name__ == '__main__':

    img_file = "../data/maths_puzzle_data.png"
    features, labels, predict_features = get_features_labes_from_image(img_file)

    print("features  " , features)
    print("labels ", labels)
    print("predict_features ", predict_features)

    #predict_features = np.array([[1,1,1]])

    #model = build_math_model(X_train=features, Y_train= labels , X_valid=features[:3], Y_valid= labels[:3])
    model = get_trained_model()

    print_weights(model)

    predicted_value = int(round(model.predict(predict_features)[0][0]))

    print("predicted_value : ", predicted_value)

    draw_image_with_predicted(img_file, predicted_value = str(predicted_value))

