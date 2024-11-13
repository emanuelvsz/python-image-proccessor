import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, add
import numpy as np
import nltk
nltk.download('punkt')

def extract_image_features(image_path):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    
    features = model.predict(image)
    return features

image_path = 'img.jpg'
image_features = extract_image_features(image_path)

def generate_caption(features, max_length=20):
    example_vocab = ["casa", "reforma", "precisa", "não", "parede", "piso", "janelas"]
    vocab_size = len(example_vocab) + 1
    word_to_index = {word: idx for idx, word in enumerate(example_vocab, 1)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    input_img_features = Input(shape=(2048,))
    img_features_reduced = Dense(256, activation='relu')(input_img_features)
    img_features_reduced = Dropout(0.5)(img_features_reduced)

    input_caption = Input(shape=(max_length,))
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
    caption_embedding = Dropout(0.5)(caption_embedding)
    caption_embedding = LSTM(256)(caption_embedding)

    decoder = add([img_features_reduced, caption_embedding])
    decoder = Dense(256, activation='relu')(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder)

    caption_model = Model(inputs=[input_img_features, input_caption], outputs=output)
    
    caption = ["<start>"]
    for i in range(max_length):
        sequence = [word_to_index[word] for word in caption if word in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = caption_model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word.get(yhat)
        
        if word is None or word == "<end>":
            break
        caption.append(word)
    
    return ' '.join(caption[1:])

print("Legenda gerada:", generate_caption(image_features))
