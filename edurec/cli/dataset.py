from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import networkx as nx
import typer
from torch_geometric.utils import to_networkx

from .. import settings
from ..datasets import DatasetName, ElearningDataModule

app = typer.Typer(no_args_is_help=True)


@app.command(name="dataset", help="Print dataset information.")
def dataset_command(
    dataset: Annotated[
        DatasetName, typer.Option("--dataset", "-d", help="Dataset to use")
    ] = DatasetName.EXPLICIT_MARS,
    max_rows: Annotated[
        int, typer.Option("--max_rows", "-m", help="Maximum number of rows to show")
    ] = 10,
) -> None:
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        test_ratio=settings.TEST_RATIO,
        val_ratio=settings.VAL_RATIO,
        use_processed_data=False,
        remove_sparse=False,
    )

    dm.prepare_data()
    dm.setup()

    print(f"Dataset name: {dataset.value}")
    print(f"Feedback type: {dm.feedback_type}")
    print(f"Split strategy: {dm.split_strategy}")
    if not dm.is_explicit:
        print(
            "Training negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive interaction"
        )
    print(f"Dataset sparsity: {dm.sparsity}")
    print(f"Number of users: {dm.num_users}")
    print(f"Number of items: {dm.num_items}")
    print(f"Number of interactions: {dm.num_interactions}")
    print(f"Number of user features: {dm.num_user_feats}")
    print(f"Number of item features: {dm.num_item_feats}")
    print(f"Number of interactions context features: {dm.num_ctx_feats}")

    raw_dataset = dm.raw_dataset
    assert raw_dataset is not None

    print(f"\nUser information:\n{raw_dataset.user_features.head(max_rows)}\n")
    print(f"Item information:\n{raw_dataset.item_features.head(max_rows)}\n")
    print(f"Interactions:\n{raw_dataset.interactions.head(max_rows)}")


@app.command(name="kg-image", help="Save a visualization of the knowledge graph.")
def kg_image_command(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output image path (PNG, SVG, etc.)")
    ] = Path("knowledge_graph.png"),
    dataset: Annotated[
        DatasetName, typer.Option("--dataset", "-d", help="Dataset to use")
    ] = DatasetName.EXPLICIT_MARS,
    max_nodes: Annotated[
        int, typer.Option("--max-nodes", min=1, help="Maximum nodes to draw")
    ] = 300,
    max_edges: Annotated[
        int, typer.Option("--max-edges", min=1, help="Maximum edges to draw")
    ] = 1000,
    seed: Annotated[int, typer.Option(help="Layout random seed")] = 42,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
) -> None:
    """Build the PyG HeteroData graph and render a bounded overview."""
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        test_ratio=settings.TEST_RATIO,
        val_ratio=settings.VAL_RATIO,
        use_processed_data=use_processed_data,
        remove_sparse=False,
    )
    dm.prepare_data()
    dm.setup()

    kg = dm.knowledge_graph
    homogeneous = kg.to_homogeneous()
    nx_graph = to_networkx(
        homogeneous,
        node_attrs=["node_type"],
        edge_attrs=["edge_type"],
        to_undirected=False,
    )

    # Keep a deterministic subset so the image remains useful for large KGs.
    nodes_by_type = {
        index: [
            node
            for node, attrs in nx_graph.nodes(data=True)
            if int(attrs["node_type"]) == index
        ]
        for index in range(len(kg.node_types))
    }
    selected_nodes: list[int] = []
    depth = 0
    while len(selected_nodes) < max_nodes:
        added = False
        for nodes in nodes_by_type.values():
            if depth < len(nodes):
                selected_nodes.append(nodes[depth])
                added = True
                if len(selected_nodes) == max_nodes:
                    break
        if not added:
            break
        depth += 1
    nx_graph = nx_graph.subgraph(selected_nodes).copy()
    if nx_graph.number_of_edges() > max_edges:
        selected_edges = (
            list(nx_graph.edges(keys=True))[:max_edges]
            if nx_graph.is_multigraph()
            else list(nx_graph.edges)[:max_edges]
        )
        limited = nx.MultiDiGraph() if nx_graph.is_multigraph() else nx.DiGraph()
        limited.add_nodes_from(nx_graph.nodes(data=True))
        for edge in selected_edges:
            source, target, *key = edge
            attrs = (
                nx_graph.get_edge_data(source, target, key[0])
                if key
                else nx_graph.get_edge_data(source, target)
            )
            limited.add_edge(source, target, **(attrs or {}))
        nx_graph = limited

    node_types = list(kg.node_types)
    edge_types = list(kg.edge_types)
    palette = plt.get_cmap("tab20")
    colors = [palette(index % 20) for index in range(len(node_types))]
    edge_colors = [palette(index % 20) for index in range(len(edge_types))]
    node_colors = [colors[int(nx_graph.nodes[node]["node_type"])] for node in nx_graph]
    positions = nx.spring_layout(
        nx_graph, seed=seed, k=1.0 / max(1, len(nx_graph)) ** 0.5
    )

    width = min(24, max(10, len(nx_graph) ** 0.5 * 0.7))
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    edge_list = (
        list(nx_graph.edges(data=True, keys=True))
        if nx_graph.is_multigraph()
        else [(*edge, data) for *edge, data in nx_graph.edges(data=True)]
    )
    nx.draw_networkx_edges(
        nx_graph,
        positions,
        ax=ax,
        edgelist=[(source, target, key) for source, target, key, _ in edge_list]
        if nx_graph.is_multigraph()
        else [(source, target) for source, target, _ in edge_list],
        edge_color=[edge_colors[int(attrs["edge_type"])] for *_, attrs in edge_list],
        alpha=0.28,
        arrows=False,
        width=0.6,
    )
    nx.draw_networkx_nodes(
        nx_graph, positions, ax=ax, node_color=node_colors, node_size=28, linewidths=0
    )
    for index, node_type in enumerate(node_types):
        if any(int(nx_graph.nodes[node]["node_type"]) == index for node in nx_graph):
            ax.scatter([], [], color=colors[index], label=node_type, s=35)
    node_legend = ax.legend(
        title="Node type", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small"
    )
    ax.add_artist(node_legend)
    edge_handles = [
        plt.Line2D([0], [0], color=edge_colors[index], label=" · ".join(edge_type))
        for index, edge_type in enumerate(edge_types)
        if any(int(attrs["edge_type"]) == index for *_, attrs in edge_list)
    ]
    if edge_handles:
        ax.legend(
            handles=edge_handles,
            title="Relation type",
            loc="lower left",
            bbox_to_anchor=(1.01, 0),
            fontsize="x-small",
        )
    ax.set_title(
        f"{dataset.value} knowledge graph "
        f"({len(nx_graph)} nodes, {nx_graph.number_of_edges()} edges)"
    )
    ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Knowledge graph image saved to {output}")
