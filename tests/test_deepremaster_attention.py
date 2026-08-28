import mlx.core as mx
import numpy as np

from src.pipeline.deepremaster_mlx import SourceReferenceAttention


def test_reference_importance_matches_dense_attention_grouping() -> None:
    attention = SourceReferenceAttention(source_channels=8, reference_channels=8)
    source = mx.random.normal((1, 2, 3, 4, 8))
    references = mx.random.normal((1, 2, 3, 2, 8))
    key, _ = attention.project_reference(references)

    actual = attention.reference_importance(source, key, reference_count=2)
    query = attention._to_tokens(attention.query(source))
    dense = mx.softmax(mx.matmul(query, mx.swapaxes(key, -1, -2)), axis=-1)
    expected = dense.reshape((1, 2, 3, 4, 2, 6)).sum(axis=-1)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual).sum(axis=-1), 1.0, atol=1e-6)
