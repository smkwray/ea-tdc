from __future__ import annotations

from ea_tdc.download import _normalize_graph_csv_payload


def test_normalize_graph_csv_payload_maps_observation_date_to_date_value() -> None:
    payload = _normalize_graph_csv_payload(
        "observation_date,SP500\n2024-01-02,4700.0\n2024-01-03,4715.2\n"
    )

    assert payload == [
        {"date": "2024-01-02", "value": "4700.0"},
        {"date": "2024-01-03", "value": "4715.2"},
    ]
