from rox_dox.model import BlockDiagram, CodeSource, Edge, Group, Node
from rox_dox.sequences import block_sequences


def _source(line: int) -> CodeSource:
    return CodeSource(path="backend/src/app.py", lines=(line, line))


def _diagram() -> BlockDiagram:
    groups = [
        Group(id="callers", label="Callers", source=_source(1)),
        Group(id="services", label="HTTP services", source=_source(2)),
        Group(id="queues", label="Queues", source=_source(3)),
        Group(id="workers", label="Workers", source=_source(4)),
        Group(id="stores", label="Stores", source=_source(5)),
    ]
    nodes = [
        Node(id="clients", label="HTTP clients", source=_source(1), group="callers"),
        Node(id="api", label="API", source=_source(2), group="services"),
        Node(id="sqs", label="SQS", source=_source(3), group="queues", kind="queue"),
        Node(id="worker", label="Worker", source=_source(4), group="workers"),
        Node(
            id="pg", label="PostgreSQL", source=_source(5), group="stores", kind="store"
        ),
    ]
    edges = [
        Edge(src="worker", dst="pg", label="reads + writes", source=_source(40)),
        Edge(src="clients", dst="api", label="REST calls", source=_source(10)),
        Edge(src="api", dst="pg", label="reads", source=_source(20)),
        Edge(src="sqs", dst="worker", label="long-poll", source=_source(30)),
    ]
    return BlockDiagram(groups=groups, nodes=nodes, edges=edges)


def test_one_sequence_per_root_in_column_order() -> None:
    sequences = block_sequences(_diagram())

    assert [sequence.title for sequence in sequences] == [
        "Flow from HTTP clients",
        "Flow from SQS",
    ]
    http, queue = sequences
    assert [participant.id for participant in http.participants] == [
        "clients",
        "api",
        "pg",
    ]
    assert [(step.src, step.dst, step.message) for step in http.steps] == [
        ("clients", "api", "REST calls"),
        ("api", "pg", "reads"),
    ]
    assert [participant.id for participant in queue.participants] == [
        "sqs",
        "worker",
        "pg",
    ]
    assert [step.source.lines for step in queue.steps] == [(30, 30), (40, 40)]


def test_shared_store_is_one_participant_with_every_incoming_step() -> None:
    diagram = _diagram()
    diagram.edges.append(
        Edge(src="api", dst="sqs", label="send_message", source=_source(21))
    )

    sequences = block_sequences(diagram)

    assert len(sequences) == 1
    (sequence,) = sequences
    assert [participant.id for participant in sequence.participants] == [
        "clients",
        "api",
        "sqs",
        "worker",
        "pg",
    ]
    assert [(step.src, step.dst) for step in sequence.steps] == [
        ("clients", "api"),
        ("api", "sqs"),
        ("api", "pg"),
        ("sqs", "worker"),
        ("worker", "pg"),
    ]


def test_edgeless_diagram_has_no_sequences() -> None:
    diagram = _diagram()
    nodes = [node for node in diagram.nodes if node.kind != "queue"]
    groups = [group for group in diagram.groups if group.id != "queues"]
    diagram = BlockDiagram(groups=groups, nodes=nodes, edges=[])

    assert block_sequences(diagram) == []
