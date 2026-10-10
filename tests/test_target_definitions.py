from separation_over_expected.target_definitions import compute_separation_targets


def _player(x, y=0.0):
    return {"x": x, "y": y}


def test_same_nearest_defender_at_both_endpoints():
    snap = {"wr": _player(0)}
    release = {"wr": _player(10)}
    snap.update({"d1": _player(3), "d2": _player(5), "d3": _player(7)})
    end = {"wr": _player(10), "d1": _player(11), "d2": _player(5), "d3": _player(18)}

    targets = compute_separation_targets(snap["wr"], release["wr"], snap, end, {"d1", "d2", "d3"})

    assert targets["nearest_defender_switched"] == "0"
    assert targets["delta_sep_endpoint_nearest"] == "-2.000"
    assert targets["delta_sep_snap_anchor"] == "-2.000"
    assert targets["delta_sep_snap_top3"] == "-2.000"


def test_nearest_defender_switch_isolated_by_snap_anchor():
    snap = {"wr": _player(0), "d1": _player(3), "d2": _player(5), "d3": _player(7), "d4": _player(12)}
    end = {"wr": _player(10), "d1": _player(4), "d2": _player(10.5), "d3": _player(13), "d4": _player(20)}

    targets = compute_separation_targets(snap["wr"], end["wr"], snap, end, {"d1", "d2", "d3", "d4"})

    assert targets["nearest_defender_switched"] == "1"
    assert targets["delta_sep_endpoint_nearest"] == "-2.500"
    assert targets["delta_sep_snap_anchor"] == "3.000"
    assert targets["delta_sep_snap_top3"] == "-2.500"


def test_top_three_allows_a_snap_nearby_defender_to_become_nearest():
    snap = {"wr": _player(0), "d1": _player(3), "d2": _player(5), "d3": _player(7), "d4": _player(12)}
    end = {"wr": _player(10), "d1": _player(15), "d2": _player(10.5), "d3": _player(13), "d4": _player(10.1)}

    targets = compute_separation_targets(snap["wr"], end["wr"], snap, end, {"d1", "d2", "d3", "d4"})

    assert targets["nearest_defender_switched"] == "1"
    assert targets["delta_sep_snap_anchor"] == "2.000"
    assert targets["delta_sep_snap_top3"] == "-2.500"
    assert targets["delta_sep_endpoint_nearest"] == "-2.900"


def test_missing_snap_anchor_at_release_marks_target_unavailable():
    snap = {"wr": _player(0), "d1": _player(3), "d2": _player(5), "d3": _player(7)}
    end = {"wr": _player(10), "d2": _player(10.5), "d3": _player(13)}

    targets = compute_separation_targets(snap["wr"], end["wr"], snap, end, {"d1", "d2", "d3"})

    assert targets["delta_sep_snap_anchor"] == ""
    assert targets["delta_sep_snap_top3"] == ""
    assert targets["snap_top3_release_count"] == "2"


def test_two_defenders_do_not_claim_a_complete_top_three_target():
    snap = {"wr": _player(0), "d1": _player(3), "d2": _player(5)}
    end = {"wr": _player(10), "d1": _player(11), "d2": _player(15)}

    targets = compute_separation_targets(snap["wr"], end["wr"], snap, end, {"d1", "d2"})

    assert targets["snap_top3_release_count"] == "2"
    assert targets["delta_sep_snap_top3"] == ""
