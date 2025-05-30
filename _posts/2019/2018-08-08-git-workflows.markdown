---
layout: post
title: Git - Workflows
date: '2018-08-08 13:19'
subtitle: Git Test Runners, CI/CD, PyPi
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - Git
---

## Local Git Repo Setup

- Copy ssh key to Github
- Set up email and username:

```python
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

## Regular Git Workflow

- Store changes of one file to stash: `git stash push FILEPATH`

## Github Test Workflow

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

## Combine A Divergence Between Remote And Local Repos

When there is a divergence between the remote and the local repos, `git pull` won't work. Instead it will show:

```bash
You have diverging branches and need to specify how to reconcile them. Before performing the next pull operation, you can suppress this message by running one of the following commands:
git config pull.rebase false  # Merge (default strategy)
git config pull.rebase true   # Rebase
git config pull.ff only       # Fast-forward only
```

To confirm that there's a divergence, we can:

1. `git log master..origin/master --oneline` to see the different commits on **remote** since the last common commit. I see

```bash
463955d (origin/master, origin/HEAD) more
f746488 more
ed0b487 more
```

2. `git log origin/master..master --oneline` to see different commits on **local** since the last common commit. I see

```bash
02af0d4 (HEAD -> master) more
ba2bb33 more
```

- Locally, this can be confirmed by `git log`. We can even have a more visual representation: `git log --graph --oneline --decorate master origin/master`

```bash
* 02af0d4 (HEAD -> master) more
* ba2bb33 more
| * 463955d (origin/master, origin/HEAD) more
| * f746488 more
| * ed0b487 more
|/  
* 89696f2 more
```

3. To fix, there are 3 options:

- `git config pull.rebase false` merge the remote branch into the local branch in a new commit.

```
A---B---C (master)
     \
      D---E (origin/master)
=>
A---B---C---F (master) (merge commit)
     \     /
      D---E (origin/master)
```

- `git config pull.rebase true` appends the local branch to the remote branch (a.k.a "rebase").

```
A---B---C (master)
     \
      D---E (origin/master)
=>
A---B---D---E---C (master) (rebased commits)
```

I like a linear history. So I do `git rebase origin/master`.

## Be Careful With Checking In Small MRs Off A Large One

- Branch 1 (b1): Contains file1, file2, and file3.

- Branch 2 (b2): Contains file2 and file3 (with modifications).

- Workflow:

    1. Merge b2 into main.

    2. Pull main back into b1, **which causes merge conflicts on file2 and file3.**

- Solution:
    1. Only alter files in Branch 1. Branch 2 is only used for merging purposes.
    2. Use Git’s `--ours` strategy to favor the changes from b1 over those in main 
        ```bash
        git fetch
        git merge -X ours main 
        ```
        

## Gitlab SSH Setup

1. At the top left corner, select `subgroup information-> group memebers`
  - Look for your username in the Project Members list. Make sure you have developer access
2. Follow this link to:

  - Generate ssh key
  - Add SSH Key to ssh-agent
  - Add SSH key to Gitlab

3. Verify connection to Gitlab using `ssh -T git@gitlab.<COMPANY>.com`

  - use `ssh-add -l` to see if you have added ssh key
  - `ssh-add ~/.ssh/id_ed25519` must come after `eval $(ssh-agent -s)`

4. If having trouble using SSH to download a repo:
  - Have a verbose command to `ssh -v git@gitlab.<COMPANY>.com`