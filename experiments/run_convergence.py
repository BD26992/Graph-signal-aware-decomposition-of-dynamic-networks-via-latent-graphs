"""Run DGD observed-objective convergence experiments."""

from __future__ import annotations

from _bootstrap import add_repo_root_to_path


add_repo_root_to_path()

from convergence_observed_curves import main


if __name__ == "__main__":
    main()

