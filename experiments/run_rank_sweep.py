"""Run DGD observed-error rank-sweep experiments."""

from __future__ import annotations

from _bootstrap import add_repo_root_to_path


add_repo_root_to_path()

from dgd_rank_sweep_observed_error import main


if __name__ == "__main__":
    main()

