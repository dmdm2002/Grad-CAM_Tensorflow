class params(object):
    def __init__(self):
        super(params, self).__init__()
        self.size = (224, 224)
        self.classes = {'fake': 0, 'live': 1}
        self.root = 'Z:/Iris_dataset/nd_labeling_iris_data/Proposed'

        self.A = f'{self.root}/1-fold/A'
        self.B = f'{self.root}/1-fold/B_blur'

        self.batchsz = 1