"""Stage 1: train a population of independent trajectories with randomized branches."""

from _common import config_from_args, parser

from mio.pipeline import stage_generate

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--workers", type=int, default=None, help="parallel processes (one root per process)")
    args = p.parse_args()
    stage_generate(config_from_args(args), args.workers)
