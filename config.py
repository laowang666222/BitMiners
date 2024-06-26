class Training_Config(object):
    def __init__(self,
                 embedding_dimension=100,
                 vocab_size=20000,
                 training_epoch=1,
                 num_val=1,
                 max_sentence_length=320,
                 cuda=True,
                 label_num=3,
                 learning_rate=1e-5,
                 batch_size=2,
                 dropout=0.3):
        self.embedding_dimension = embedding_dimension  # 词向量的维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.epoch = training_epoch  # 训练轮数
        self.num_val = num_val  # 经过几轮才开始验证
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率


