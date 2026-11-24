# Publish Rstats to PyPI

The distribution name is **rstatspython**; Python code still uses
`import rstats as rs`. The name was unregistered at the time of preparation;
only a successful first upload claims it.

## One-time owner setup

1. Sign in to [PyPI](https://pypi.org/account/login/), or create an account and
   finish email verification and two-factor setup if prompted.
2. Open [your Publishing settings](https://pypi.org/manage/account/publishing/).
3. Under **Add a new pending publisher**, select **GitHub** and enter:

| PyPI field | Value |
| --- | --- |
| PyPI project name | `rstatspython` |
| Owner | `drdanrb` |
| Repository name | `Rstats` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

4. Click **Add**. A pending publisher creates the project on its first upload;
   it does not reserve the project name in advance.

No PyPI API token or password belongs in this repository. The publishing job
uses GitHub's short-lived OIDC identity, matched to the fields above. See
[PyPI's pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Publish the first version

After the owner setup, open the repository's
[Publish to PyPI workflow](https://github.com/drdanrb/Rstats/actions/workflows/publish.yml),
choose **Run workflow**, select `main`, enable **publish**, and run it.
Or use the authenticated GitHub CLI:

```bash
gh workflow run publish.yml --repo drdanrb/Rstats --ref main -f publish=true
```

The workflow tests the library, builds the source distribution and wheel,
validates package metadata with Twine, then uploads using the `pypi` environment.
Leaving **publish** disabled performs build/validation only.

After a successful upload, verify a fresh installation:

```bash
python -m pip install rstatspython==0.1.0
python -c "import rstats as rs; print(rs.__version__)"
```

Then update the README/notebook installation examples from GitHub to
`pip install rstatspython`. Keep the GitHub fallback until the registry install
has actually been verified.

## Later versions

Update `project.version` in `pyproject.toml` and `rstats.__version__` together.
Publish a GitHub release with the matching tag, e.g. `v0.1.1`. Published GitHub
releases trigger the same workflow, and a tag/version mismatch stops the build.
PyPI versions cannot be overwritten; bump the version for a new upload. The
workflow deliberately does not silently skip existing files.

## Anaconda users

Students can use the PyPI package with pip inside an activated conda environment.
A native `conda install` package requires a separate Anaconda channel or
conda-forge recipe, builds, and maintenance. PyPI is the initial publishing
target; no native conda package is claimed.
