

class Client():
    def __init__(self, train_data, test_data, batch_size, model, loc_steps, data_size):
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.model = model
        self.loc_steps = loc_steps
        self.data_size = data_size

        self.ba = None
        self.global_steps = 0
        self.have_trained = False
    def set_ba(self, ba):
        self.ba = ba

    def precheck(self):
        if self.ba is None:
            return True
        else:
            return self.ba.precheck(self.data_size, self.batch_size, self.loc_steps)

