from .neural_qcfg import NeuralQCFGTgtParser
from .struct.pcfg import PCFG
from .struct.pcfg_rdp import PCFGRandomizedDP


class NeuralQCFGRDPTgtParser(NeuralQCFGTgtParser):
    def __init__(self, rdp_topk=5, rdp_sample_size=5, rdp_smooth=0.001, **kwargs):
        super(NeuralQCFGRDPTgtParser, self).__init__(**kwargs)
        self.pcfg_inside = PCFGRandomizedDP(rdp_topk, rdp_sample_size, rdp_smooth)
        self.pcfg_decode = PCFG()

    def forward(self, *args, **kwargs):
        self.pcfg = self.pcfg_inside
        return super().forward(*args, **kwargs)

    def parse(self, *args, **kwargs):
        self.pcfg = self.pcfg_decode
        return super().parse(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.pcfg = self.pcfg_decode
        return super().generate(*args, **kwargs)
