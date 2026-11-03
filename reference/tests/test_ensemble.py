import numpy as np
import pytest

from reference.ensemble import DAVIS_SEEDS, MemberPredictions, aggregate_davis


def records(scores=None, pair_ids=("davis:9", "davis:2")):
    scores = ([6., 8.], [8., 9.], [10., 10.]) if scores is None else scores
    return [MemberPredictions(seed, pair_ids, values) for seed, values in zip(DAVIS_SEEDS, scores)]


def test_arithmetic_mean_is_in_pkd_space_and_sd_is_descriptive_sample_sd():
    members = records()
    result = aggregate_davis(members[::-1])
    assert result.columns.tolist() == ["pair_id", *(f"member_{seed}_pKd" for seed in DAVIS_SEEDS),
                                        "mean_pKd", "member_SD_pKd"]
    assert result.pair_id.tolist() == ["davis:9", "davis:2"]
    np.testing.assert_array_equal(result.mean_pKd, [8., 9.])
    np.testing.assert_array_equal(result.member_SD_pKd, [2., 1.])
    transformed_mean_kd = -np.log10(np.mean(10. ** -np.array([6., 8., 10.])))
    assert result.mean_pKd.iloc[0] != pytest.approx(transformed_mean_kd)


def test_member_inputs_and_original_order_are_not_mutated_or_shared():
    ids = ["davis:9", "davis:2"]
    values = [np.array([6., 8.]), np.array([8., 9.]), np.array([10., 10.])]
    originals = [array.copy() for array in values]
    members = records(values, ids)
    result = aggregate_davis(members)
    result.loc[0, "member_20261001_pKd"] = -1.
    assert ids == ["davis:9", "davis:2"]
    for original, actual in zip(originals, values):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("ids", [("davis:2", "davis:9"), ("davis:9", "other")])
def test_mismatched_observation_order_or_identity_is_rejected(ids):
    members = records()
    members[1] = MemberPredictions(DAVIS_SEEDS[1], ids, [8., 9.])
    with pytest.raises(ValueError, match="ordering"):
        aggregate_davis(members)


@pytest.mark.parametrize("ids", [("a", "a"), ("a", ""), (), (1, 2)])
def test_invalid_or_duplicate_observation_identifiers_are_rejected(ids):
    with pytest.raises(ValueError, match="Pair IDs"):
        aggregate_davis(records(pair_ids=ids))


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_exactly_three_members_are_required(count):
    members = records()
    with pytest.raises(ValueError, match="exactly three members"):
        aggregate_davis((members * 2)[:count])


def test_repeated_or_undeclared_seeds_are_rejected():
    members = records()
    with pytest.raises(ValueError, match="distinct"):
        aggregate_davis([members[0], members[0], members[2]])
    with pytest.raises(ValueError, match="declared ensemble"):
        aggregate_davis([members[0], members[1], MemberPredictions(7, members[2].pair_ids, [10., 10.])])
    alternate = [MemberPredictions(seed, member.pair_ids, member.scores)
                 for seed, member in zip([3, 1, 2], members)]
    assert aggregate_davis(alternate, expected_seeds=[3, 2, 1]).columns.tolist()[1:4] == [
        "member_1_pKd", "member_2_pKd", "member_3_pKd"]
    with pytest.raises(ValueError, match="three distinct"):
        aggregate_davis(members, expected_seeds=[1, 1, 2])


@pytest.mark.parametrize("scale", ["KIBA_score", "pKi", "pIC50", "pX", "Kd", ""])
def test_untrained_endpoint_heads_and_other_scales_are_rejected(scale):
    members = records()
    members[1] = MemberPredictions(DAVIS_SEEDS[1], members[1].pair_ids, [8., 9.], scale)
    with pytest.raises(ValueError, match="pKd"):
        aggregate_davis(members)


@pytest.mark.parametrize("bad", [[np.nan, 1.], [np.inf, 1.], [-np.inf, 1.], [[1., 2.]], [1.],
                                [True, False], ["1", "2"], [1j, 2j]])
def test_nonfinite_misaligned_or_nonnumeric_scores_are_rejected(bad):
    members = records()
    members[0] = MemberPredictions(DAVIS_SEEDS[0], members[0].pair_ids, bad)
    with pytest.raises(ValueError):
        aggregate_davis(members)


def test_noninteger_seeds_are_rejected():
    members = records()
    for bad in [True, "20261001", 20261001.0]:
        with pytest.raises(ValueError, match="integer"):
            aggregate_davis([MemberPredictions(bad, members[0].pair_ids, members[0].scores), *members[1:]])
