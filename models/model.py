class ModelNotTrainedException(Exception):
    pass

class Model(object):
    def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y

        self.is_trained = False

        self.predictions = None
        self.errors = {}
        self.metadata = {}

    def is_trained(self):
        return self.is_trained

    def train(self):
        self.is_trained = True
        self._train()

    def _train(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def get_predictions(self):
        if not self.is_trained:
            raise ModelNotTrainedException
        else:
            return self._get_predictions()

    def _get_predictions(self):
        return self.predictions

    def get_errors(self):
        if not self.is_trained:
            raise ModelNotTrainedException
        else:
            return self._get_errors()

    def _get_errors(self):
        return self.errors

    def get_metadata(self):
        if not self.is_trained:
            raise ModelNotTrainedException
        else:
            return self._get_metadata()

    def _get_metadata(self):
        return self.metadata
