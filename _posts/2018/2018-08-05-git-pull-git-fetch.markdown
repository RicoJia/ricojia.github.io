---
layout: post
title: Git Pull And Fetch
date: 2018-08-08 13:19
subtitle:
comments: true
tags:
  - Linux
---
## Git Fetch 

 `git fetch <REMOTE_MACHE>:<DIR> <BRANCH_NAME>`


## git pull

git pull = git fetch + one of two ways to reconcile:

- merge (the default) — creates a merge commit joining your local commits and the new remote commits.
- rebase — replays your local commits on top of the fetched remote tip, so history stays linear (no merge commit).

Without either preference set, git pull refuses and shows you the exact message you got: "You have divergent branches and need to specify how to reconcile them."

`git config pull.rebase true`  Makes every future git pull in this repo behave as git pull --rebase automatically — no more prompt, and no merge commits from routine pulls.
