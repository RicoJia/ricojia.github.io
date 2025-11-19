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

---

## Regular Git Workflow

- Store changes of one file to stash: `git stash push FILEPATH`


Here is the cleaned-up, properly formatted **raw Markdown** (no code fences around the whole thing—**this is the final raw markdown**):

---

## GitHub Test Workflow

Test workflow is done through GitHub Actions. Here we are going to create a workflow on GitHub's default runner.

### Steps

* Create a `.github/workflows/test.yml`
* Add a custom Docker file `Docker_test_container` in the root directory
* In the **Actions** tab on your repo's panel, you can check test runs
* For a full example, please see here: **TODO**

**Note:** `pre-commit` can be used to run tests even before committing.

---

### 1. Install `pre-commit`

```bash
pip install pre-commit
```

### 2. Create a `.pre-commit-config.yaml` file in your project’s root directory:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0  # Use the latest version of black
    hooks:
      - id: black
```

### 3. Install the pre-commit hooks

```bash
pre-commit install
```

---

## Enforce Tests Before Merging (Branch Protection Rules)

To ensure tests pass before a pull request can be merged:

* Go to:
  **Settings → Branches → Branch protection rules → Add rule**

  * In **Branch name pattern**, enter `master` (or the protected branch name)
  * Check **Require status checks to pass before merging**
  * Optionally enable additional rules (e.g., *Require approvals*)

From the list of available checks, select the GitHub Actions workflow (usually named after your workflow file, such as **Run Tests**).

---

## Add a Status Badge

You can add a status badge that **automatically updates** based on your workflow status:

1. Go to **Actions**
2. Click on the workflow (e.g., *Run Tests*)
3. Click the **...** (three dots) button → **Create status badge**
4. GitHub will generate a Markdown snippet, such as:

```markdown
![example workflow](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/test.yml/badge.svg)
```

5. Copy it into your `README.md`:

```markdown
# My Project
![Run Tests](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/test.yml/badge.svg)

This is my project description...
```

---

## Compare Branches

To compare branches:

* **GitLab**: Use the right-hand side panel for *Compare Revisions*

---


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

Here is the **cleaned, formatted, and tidy raw Markdown** for all sections you provided.

---

## Combine a Divergence Between Remote and Local Repos

When there is a divergence between the remote and local repositories, a simple `git pull` will fail with:

```bash
You have diverging branches and need to specify how to reconcile them. Before performing the next pull operation, you can suppress this message by running one of the following commands:
git config pull.rebase false  # Merge (default strategy)
git config pull.rebase true   # Rebase
git config pull.ff only       # Fast-forward only
```

### Confirming the Divergence

1. View commits on **remote** since the last common base:

   ```bash
   git log master..origin/master --oneline
   ```

   Example:

   ```bash
   463955d (origin/master, origin/HEAD) more
   f746488 more
   ed0b487 more
   ```

2. View commits on **local** since the last common base:

   ```bash
   git log origin/master..master --oneline
   ```

   Example:

   ```bash
   02af0d4 (HEAD -> master) more
   ba2bb33 more
   ```

3. Visualize both histories together:

   ```bash
   git log --graph --oneline --decorate master origin/master
   ```

   Example:

   ```bash
   * 02af0d4 (HEAD -> master) more
   * ba2bb33 more
   | * 463955d (origin/master, origin/HEAD) more
   | * f746488 more
   | * ed0b487 more
   |/  
   * 89696f2 more
   ```

### How to Fix the Divergence

There are three options:

#### **1. Merge (default)**

`git config pull.rebase false`
This merges the remote branch into the local one and creates a merge commit.

```
A---B---C (master)
     \
      D---E (origin/master)
=>
A---B---C---F (master)
     \     /
      D---E (origin/master)
```

#### **2. Rebase**

`git config pull.rebase true`
This replays local commits on top of the remote branch.

```
A---B---C (master)
     \
      D---E (origin/master)
=>
A---B---D---E---C (master)
```

If you prefer a linear history (like I do), run:

```bash
git rebase origin/master
```

---

## Be Careful When Checking In Small MRs Based on a Large MR

**Scenario:**

* **Branch 1 (b1):** Has `file1`, `file2`, and `file3`.
* **Branch 2 (b2):** Has changes only to `file2` and `file3`.

**Workflow:**

1. Merge **b2** into `main`.
2. Pull `main` back into **b1** → merge conflicts occur on `file2` and `file3`.

**Solution:**

1. Only edit files in **b1**. Use b2 solely for merging purposes.
2. Use Git’s `--ours` strategy to favor **b1** changes during merge:

   ```bash
   git fetch
   git merge -X ours main
   ```

---

## GitLab SSH Setup

1. In GitLab, navigate to:
   **Subgroup information → Group members**

   * Find your username in the Project Members list
   * Ensure you have **Developer** access

2. Follow GitLab documentation to:

   * Generate an SSH key
   * Add the SSH key to `ssh-agent`
   * Add the SSH key to GitLab

3. Verify your SSH connection:

   ```bash
   ssh -T git@gitlab.<COMPANY>.com
   ```

   Helpful checks:

   * List loaded keys:

     ```bash
     ssh-add -l
     ```

   * Add key (must be done *after* starting the agent):

     ```bash
     eval $(ssh-agent -s)
     ssh-add ~/.ssh/id_ed25519
     ```

4. If you still have trouble cloning or pulling via SSH:

   ```bash
   ssh -v git@gitlab.<COMPANY>.com
   ```

   (Verbose output helps identify issues.)

---

If you'd like, I can combine everything into a single README or cheat-sheet page.



