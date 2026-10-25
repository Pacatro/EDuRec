"""Build the proposed model's graph and query datasets from canonical data."""

from functools import cached_property

from torch_geometric.data import HeteroData

from edurec.datasets import graphs
from edurec.datasets.adapters.base import ModelDataAdapter
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery


class ProposedAdapter(ModelDataAdapter[RecSysQuery, RecSysQuery]):
    @cached_property
    def knowledge_graph(self) -> HeteroData:
        """Static item knowledge graph, built once and treated as read-only."""
        return graphs.build_knowledge_graph(
            self.data.item_features,
            self.data.i_static_feats,
            self.data.schema["items"],
            self.data.graph_node_features,
        )

    def dataset(self, split: str) -> RecSysDataset:
        return self.get_queries(split)

    def evaluation_input(self, query: RecSysQuery) -> RecSysQuery:
        return query
