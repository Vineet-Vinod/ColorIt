.PHONY: verify-env download-weights colorize-movie

verify-env:
	uv run colorit verify-env

download-weights:
	uv run colorit download-weights

colorize-movie:
	uv run colorit colorize-movie
