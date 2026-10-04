"""Stage 2: build the transition index (source -> target records) from a generated population."""

import json

from _common import config_from_args, parser

from mio.pipeline import stage_transitions

if __name__ == "__main__":
    summary = stage_transitions(config_from_args(parser(__doc__).parse_args()))
    print(json.dumps(summary, indent=2))
