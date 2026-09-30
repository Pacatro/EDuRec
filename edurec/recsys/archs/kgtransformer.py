from torch import nn

from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.archs.modules.seq_encoder import TransformerSeqEncoder
from edurec.recsys.configs import ModelConfig


class KGTransformer(KGSeq):
    """Knowledge-graph educational recommender with a Transformer sequence encoder.

    Identical to :class:`~edurec.recsys.archs.kgseq.KGSeq` except that each
    user's chronological history is encoded by self-attention over learned
    positional embeddings instead of a recurrent network.
    """

    def _build_seq_encoder(self, cfg: ModelConfig) -> nn.Module:
        return TransformerSeqEncoder(cfg.transformer_seq_encoder)
