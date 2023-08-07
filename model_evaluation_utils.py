from plot_learning_callback import PlotLearning
from preprocessing import init_data_generator


# def plot_hist(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(acc) + 1)
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()

#     plt.figure()

#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()

#     plt.show()


def training_evaluation(model, epochs):
    train_generator, validation_generator, test_generator = init_data_generator()
    history = model.fit(
        train_generator, validation_data=validation_generator, epochs=epochs, verbose=0)
    results = model.evaluate(test_generator, batch_size=32)
    # history = model.fit(
    #     train_generator, validation_data=validation_generator, epochs=epochs,
    #     callbacks=[PlotLearning()])
    return history, model, results
