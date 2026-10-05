"""Run repeated real-data missing-link experiments."""

from __future__ import annotations

from _bootstrap import add_repo_root_to_path


add_repo_root_to_path()

from real_repeated_mask_experiments import main


if __name__ == "__main__":
    main()

