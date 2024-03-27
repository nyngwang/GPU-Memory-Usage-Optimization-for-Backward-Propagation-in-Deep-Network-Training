

class FakeCmdargs:
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.out_classes = 1000
        self.data = '/ssd/imagenet'
        # specs variation.
        self.sbc = False
        self.bn = False
        # checkpointing.
        self.gc = True
        self.smd = 'None'
        self.cp = None


