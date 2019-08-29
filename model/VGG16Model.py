import numpy as np
# from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
# from keras.applications.vgg19 import preprocess_input, VGG19, decode_predictions
from keras.applications.resnet import preprocess_input, ResNet152, decode_predictions


# x = image_data_format()
# from keras.applications import MobileNetV2
from keras.preprocessing.image import array_to_img, load_img, img_to_array


class VGG16Model:
    # Process Model
    # model = VGG19(
    model = ResNet152(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000
    )

    # model = MobileNetV2(weights='imagenet', include_top=False)

    def infern(self, img: np.ndarray):
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)

        predictions = self.model.predict(x)
        return decode_predictions(predictions, top=3)

    def test(self):
        # image = load_img('../4051378654_238ca94313.jpg', target_size=(224, 224))
        image = load_img('../1480654305.jpg', target_size=(224, 224))
        image = img_to_array(image)
        print(self.infern(image))

if __name__ == "__main__":
    VGG16Model().test()