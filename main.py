import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

#setamos o seed para reprodução do experimento
np.random.seed(2)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#retiramos a informação do digito
x_train = df_train.drop(["label"], axis=1).values

#apesar do dataset ja estar no formato 28x28, o framework do keras espera que seja
#informado a terceira dimensão,portanto já redimensionamentos para 28x28x1.
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = df_test.values.reshape((df_test.shape[0], 28, 28, 1))

# utilizamos a função to_categorial do utils do keras para fazermos o one-hot-encoder da classe.
y_train = df_train["label"].values
y_train = to_categorical(y_train)

#visualizando randomicamente algumas imagens
for i in range(0, 6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    plt.subplot(3,2,i+1)
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplots_adjust(top=1.4)
plt.savefig('digitos.png')
plt.show()


x_train = x_train / 255
x_test = x_test / 255
model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28,1)))
model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

    # reduz o parâmetro de learning rate se não houver 
    # melhoras em determinado número de epocas
    # útil para encontrar o mínimo global.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

batch_size = 32
epochs = 10

history = model.fit(x_train,
                            y_train,
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_split=0.2,
                            verbose = 1,
                            callbacks=[learning_rate_reduction])

history_dict = history.history
print(history_dict)
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
range_epochs = range(1, len(acc) + 1)

plt.style.use('default')
accuracy_val = plt.plot(range_epochs, val_acc, label='Acurácia no conjunto de validação')
accuracy_train = plt.plot(range_epochs, acc, label='Acurácia no conjunto de treino', color="r")
plt.setp(accuracy_val, linewidth=2.0)
plt.setp(accuracy_train, linewidth=2.0)
plt.xlabel('Épocas') 
plt.ylabel('Acurácia')
plt.legend(loc="lower right")
plt.savefig('cnn.png')
plt.show()


predictions_proba = model.predict(x_test)
predictions = np.argmax(predictions_proba, axis=1)

plt.figure(figsize=(7,14))
for i in range(0, 8):
    random_num = np.random.randint(0, len(x_test))
    img = x_test[random_num]
    plt.subplot(6,4,i+1)
    plt.margins(x = 20, y = 20)
    plt.title('Predição: ' + str(predictions[random_num]))
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.savefig('final.png')

submission = pd.DataFrame({'ImageID' : pd.Series(range(1,28001)), 'Label' : predictions})
submission.to_csv("submission.csv",index=False)
