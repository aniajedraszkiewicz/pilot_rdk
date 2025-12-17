import csv

from experiment_files.block import Block


def test_append_log_row_creates_and_appends(tmp_path):
    """
    Block.append_log_row() should:
    1) create the CSV (write header) on first call
    2) append rows on subsequent calls (no overwrite)
    """

    csv_path = tmp_path / "results.csv"
    header = ["a", "b", "c"]
    row1 = {"a": 1, "b": 2, "c": 3}
    row2 = {"a": 4, "b": 5, "c": 6}

    block = Block(
        win=None,
        kb=None,
        rdk=None,
        block_no=1,
        subject_id="S1",
        results_csv_path=str(csv_path),
        results_header=header,
        max_stim_sec=5.0,
        debug=False,
    )

    block.append_log_row(str(csv_path), row1, header)
    block.append_log_row(str(csv_path), row2, header)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2

    assert rows[0]["a"] == "1"
    assert rows[0]["b"] == "2"
    assert rows[0]["c"] == "3"

    assert rows[1]["a"] == "4"
    assert rows[1]["b"] == "5"
    assert rows[1]["c"] == "6"

    # Best header check: compare parsed fieldnames (order preserved)
    assert reader.fieldnames == header
