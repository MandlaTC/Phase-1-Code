from .preprocessing import init_data_generator


def training_evaluation(model, epochs):
    train_generator, validation_generator, test_generator = init_data_generator()
    history = model.fit(
        train_generator, validation_data=validation_generator, epochs=epochs, verbose=0)
    results = model.evaluate(test_generator, batch_size=32)
    return history, model, results
