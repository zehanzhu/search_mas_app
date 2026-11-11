from pathlib import Path

from search_mas.apps.search.data_utils import load_records, record_to_sample

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_test_sampled_parquet_parses() -> None:
    records = load_records(REPO_ROOT / "test_sampled.parquet")
    assert len(records) == 210

    sample = record_to_sample(records[0], 0)
    assert sample.sample_id == "nq-test-0"
    assert sample.data_source == "nq"
    assert sample.question == "who got the first nobel prize in physics?"
    assert sample.expected_answers == ["Wilhelm Conrad Röntgen"]
