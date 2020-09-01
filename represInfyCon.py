import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()

data= "Me gustas cuando callas porque estás como ausente, \n y me oyes desde lejos, y mi voz no te toca. \n Parece que los ojos se te hubieran volado \n y parece que un beso te cerrara la boca. \n Como todas las cosas están llenas de mi alma \n emerges de las cosas, llena del alma mía. \n Mariposa de sueño, te pareces a mi alma, \n y te pareces a la palabra melancolía. \n Me gustas cuando callas y estás como distante. \n Y estás como quejándote, mariposa en arrullo. \n Y me oyes desde lejos, y mi voz no te alcanza: \n déjame que me calle con el silencio tuyo. \n Déjame que te hable también con tu silencio \n claro como una lámpara, simple como un anillo. \n Eres como la noche, callada y constelada. \n Tu silencio es de estrella, tan lejano y sencillo. \n Me gustas cuando callas porque estás como ausente. \n Distante y dolorosa como si hubieras muerto. \n Una palabra entonces, una sonrisa bastan. \n Y estoy alegre, alegre de que no sea cierto."

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print("Representación de la información \n")
print("y me oyes desde lejos, y mi voz no te toca. \n")
print(tokenizer.word_index['y'])
print(tokenizer.word_index['me'])
print(tokenizer.word_index['oyes'])
print(tokenizer.word_index['desde'])
print(tokenizer.word_index['lejos'])
print(tokenizer.word_index['y'])
print(tokenizer.word_index['mi'])
print(tokenizer.word_index['voz'])
print(tokenizer.word_index['no'])
print(tokenizer.word_index['te'])
print(tokenizer.word_index['toca'])
print(xs[15])
print(ys[15])
print(xs[16])
print(ys[16])

input("Continuar")

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=270, verbose=1)

print("Representación del conocimiento \n")
seed_text = "Como todas las cosas"
next_words = 10
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
