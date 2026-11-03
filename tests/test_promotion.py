from types import SimpleNamespace

from src.food11.promote import gate_candidate


class FakeClient:
    def __init__(self, candidate_accuracy):
        self.candidate_accuracy = candidate_accuracy
        self.aliases = {}
        self.tags = {}

    def get_model_version(self, name, version):
        return SimpleNamespace(run_id=f"candidate-{version}")

    def get_model_version_by_alias(self, name, alias):
        assert alias == "champion"
        return SimpleNamespace(run_id="champion-run")

    def get_run(self, run_id):
        accuracy = 0.60 if run_id == "champion-run" else self.candidate_accuracy
        return SimpleNamespace(data=SimpleNamespace(metrics={"test_accuracy": accuracy}))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[key] = value


def test_gate_promotes_clear_improvement():
    client = FakeClient(0.90)
    result = gate_candidate(client, "food11", "8", 100)
    assert result["decision"] == "champion"
    assert client.aliases == {"champion": "8"}
    assert client.tags["gate_decision"] == "champion"


def test_gate_keeps_weak_candidate_as_challenger():
    client = FakeClient(0.65)
    result = gate_candidate(client, "food11", "9", 100)
    assert result["decision"] == "challenger"
    assert client.aliases == {"challenger": "9"}
    assert client.tags["gate_decision"] == "challenger"
