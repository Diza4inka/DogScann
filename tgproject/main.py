import telebot
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Замените на ваш токен
bot = telebot.TeleBot('8294896392:AAGRqCdVGDuyyx3i1qJ-OIIvWsQO_wF1sFw')

def predict_image_class(image_path, model_path, labels_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    with open(labels_path, "r", encoding='utf-8') as f:
        class_names = f.readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Удаляем лишние пробелы
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот который поможет тебе определить породу собаки. Отправь картинку, и я скажу тебе породу!")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('dog_image.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    model_path = "keras_Model.h5"
    labels_path = "labels.txt"
    class_name, confidence_score = predict_image_class('dog_image.jpg', model_path, labels_path)
    response_message = f'Похоже, это порода под номером {class_name}!'
    bot.reply_to(message, response_message)

if __name__ == '__main__':
    bot.polling(none_stop=True)