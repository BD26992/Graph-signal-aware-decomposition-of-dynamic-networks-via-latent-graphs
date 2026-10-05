"""Run repeated SwDyn missing-link experiments."""

from __future__ import annotations

from _bootstrap import add_repo_root_to_path


add_repo_root_to_path()

from swdyn_repeated_mask_experiments import main


if __name__ == "__main__":
    main()

