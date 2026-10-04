"""Stage 3: train the learned improvement operator on training-root transitions."""

from _common import config_from_args, parser

from mio.pipeline import stage_train_operator

if __name__ == "__main__":
    stage_train_operator(config_from_args(parser(__doc__).parse_args()))
