__all__ = ['Configuration', 'config']


class Configuration:
    def __init__(self):
        # #data
        self.data_dir = '/data/lf/graduate_projects/dataset'    # 绝对路径，防止同步时的大量数据传输
        self.ckpt_dir = 'checkpoint'
        self.log_dir = 'log'
        self.experiment_dir = 'experiments'

        # #model
        self.emb_size = 16
        self.T = 3  # 模型超参数T

        # #train
        # supervised learning
        self.lr = 1e-3
        self.num_epoch = 5
        self.clip_norm = 5.
        self.batch_size = 16
        self.retrain = True
        self.save_every = 20
        self.beta = 1.2  # 正类损失权重

        # reinforcement learning
        self.memory_size = 10000
        self.epsilon = 0.1
        self.gama = 0.9

        # #infer
        self.threshold = 0.5

        # #device
        self.gpu_id = 1

    def parse_args(self, kwargs):
        for key, value in kwargs.items():
            if key not in self.__dict__:
                raise ValueError('UnKnown Config "--%s"' % key)
            else:
                setattr(self, key, value)

        # 打印参数
        print('=================user config===================')
        for key, value in self.__dict__.items():
            print("%20s:\t%s" % (key, value))
        print('=====================end=======================')

    def state_dict(self):
        return {key: getattr(self, key) for key in self.__dict__}


config = Configuration()
