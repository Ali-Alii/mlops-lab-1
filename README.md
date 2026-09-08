# MLOps Lab: uv, DVC, and Food-11

This project uses [uv](https://docs.astral.sh/uv/) for its Python environment and
[DVC](https://dvc.org/) for versioning the Food-11 data. Prepare the images with:

```powershell
uv run python ./src/food11/data.py
```

## Lab answers

1. `uv init` creates `pyproject.toml` (project metadata and dependencies),
   `.python-version` (the selected Python version), `README.md`, and a starter
   `main.py`. `uv add` also creates `uv.lock`, which pins exact dependency versions,
   and `.venv`, the local environment. Commit the metadata and lockfile; ignore the
   environment and uv cache.

2. `dvc init` creates `.dvc/config` (repository-level DVC configuration),
   `.dvc/.gitignore` (keeps DVC cache/temp state out of Git), and `.dvcignore`
   (patterns DVC itself should skip). Commit `.dvc/config`, `.dvc/.gitignore`, and
   `.dvcignore`; never commit `.dvc/cache` or `.dvc/tmp`.

3. With `--global`, DVC credentials are stored in the user's global DVC config,
   outside the repository. Other levels are `--system`, `--local`, and the default
   repository config. Secrets should use `--local` or `--global`, never the tracked
   repository config, and must not be pushed to GitHub.

4. `dvc add data` adds `/data` to `.gitignore`, because the actual data is now
   stored in DVC's content-addressed cache/remote rather than Git.

5. `data.dvc` is a small YAML pointer. It records the tracked output path, content
   hash, size, and file count, allowing DVC to identify the exact dataset version.

6. GitHub contains code and `data.dvc`, but not the image files. `data.dvc` points
   indirectly to their content hashes. After `dvc push`, DagsHub's DVC storage
   contains the actual data objects and its UI can display the versioned dataset.

7. A fresh Git clone contains `data.dvc`, but the ignored `data` directory may be
   absent. Run `uv sync` followed by `uv run dvc pull` to download and check out it.

8. Checking out the earlier Git commit and running `dvc checkout` restores the
   dataset referenced by that commit, so later `food11_processed` and
   `food11_processed_mini` folders disappear. Returning to `main` and running
   `dvc checkout` restores them.
