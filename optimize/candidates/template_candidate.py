from __future__ import annotations


CANDIDATE_NAME = "template_candidate"


def is_supported(case: dict) -> bool:
    return False


def prepare_case(case: dict, device):
    return None


def run_case(case: dict, x, weight, bias, *, prepared_state=None):
    raise NotImplementedError(
        "Implement run_case(...) in a copied candidate module before benchmarking."
    )
