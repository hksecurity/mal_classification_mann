

class init_value:
    def __init__(self):
        self.n_classes = 5
        self.seq_length = 50  # time
        self.read_head_num = 4
        self.batch_size = 16
        self.insts_size = 400
        self.num_epoches = 100000
        self.learning_rate = 1e-3
        self.rnn_size = 200
        self.rnn_num_layers = 1
        self.memory_size = 128
        self.memory_vector_dim = 40
        self.model_dir = 'model'
        self.tensorboard_dir = 'tensorboard'
        self.celltype = 'mann'
        # self.celltype = 'lstm'

        self.image_width = 20
        self.image_height = 20
        # mal60
        # self.n_train_classes = 45
        # self.n_test_classes = 15
        # malimg
        self.n_train_classes = 16
        self.n_test_classes = 9