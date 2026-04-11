.PHONY: verify-env download-weights

verify-env:
	uv run colorit verify-env

download-weights:
	uv run colorit download-weights
