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
TODO

- In the `Actions` Tab on your repo's panel, you can check test runs

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

```python
# My Project

![Run Tests](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/test.yml/badge.svg)

This is my project description...
```