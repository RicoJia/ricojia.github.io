---
layout: post
title: Git - Workflows
date: '2019-09-10 13:19'
subtitle: Git Test Runners
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - Git
---

## Set Up Github Test Workflow

Test Workflow is done through Github Actions. Here we are going to create a workflow on Github's default runner

- Create a `.github/workflows/test.yml`
- Add a custom docker file `Docker_test_container`  in the root directory
- In the `Actions` Tab on your repo's panel, you can check test runs
- For a full example, please see here: TODO

Note: `pre-commit` can be used to run tests even before commiting.

1. `pip install pre-commit`
1. Create a .pre-commit-config.yaml file in your project’s root directory with the following content:

```markdown
repos:
- repo: https://github.com/psf/black
    rev: 23.1.0  # Use the latest version of black
    hooks:
    - id: black
```

1. Run the following command to install the pre-commit hooks, so black will run automatically before commit and fix formatting issues:

```bash
pre-commit install
```

To enforce that tests pass before allowing a pull request to be merged, you can enable branch protection rules:

- `Settings` -> `Branches` -> `Branch protection rules` -> `Add rule`.
  - In the "Branch name pattern" field, enter master (or the name of the branch you want to protect).
  - Check the option "Require status checks to pass before merging."
    - You can set other rules such as "require approvals"

From the list of available checks, select the one that corresponds to your GitHub Actions test (usually named after your workflow file, such as Run Tests).

Additionally, we can add a status badge, which **dynamically updates** based on the current status of your workflow:

- Go to `Actions`
- Click on the workflow that you want to create a badge for (e.g., Run Tests)
- Near the top right corner of the workflow's page, you'll see a ... (three dots) button. Click it. Click on Create status badge.
- GitHub will generate a Markdown snippet for your badge.  It will look something like this:

```markdown
![example workflow](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/test.yml/badge.svg)
```

- Copy the Markdown snippet and open your repository's README.md file.

```markdown
# My Project
![Run Tests](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/test.yml/badge.svg)
This is my project description...
```

## Set Up Github Pypi Workflow

- Add secrets to repo:
    1. `Settings` -> `Secrets and variables` (left side bar) -> `Actions` -> `New repository secret`
    2. Add secret
        - "name" field: `PYPI_USERNAME`, `PYPI_PASSWORD`
        - "secret" field: secret value

- Create a workflow file `.github/workflows/your-workflow.yml`

```yaml
name: Publish to PyPI
on:
  push:
    tags:
      - 'v*'
jobs:
  deploy:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ['3.8', '3.9', '3.10']
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: |
        python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        python -m twine upload dist/*
```

When you want to release a new version, update the version number in your `setup.py` or `pyproject.toml`, then commit and push a new tag to master:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This will trigger the publish workflow and automatically push the package to PyPI.

- Trigger when a new tag starting with v (e.g., v1.0.0) is pushed to the master branch.
- Build the package using build and twine.
- Publish the package to PyPI.
