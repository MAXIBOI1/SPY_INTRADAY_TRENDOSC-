# Git: New Repository and New Branch Process

Use this when you want to **(A)** create a new GitHub repo from a local folder, or **(B)** create a new branch in an existing repo. Paste this file (or its contents) into Cursor when you need it.

---

## When to use this

- **New repo:** You have a project folder (with or without existing code) and want it as its own GitHub repository with a specific name.
- **New branch:** You're already in a Git repo and want a new branch (e.g. `feature/xyz` or `experiment/abc`).

---

## A. Create a new GitHub repository from a local folder

**Prerequisites:** The folder is your project root. You want it to become a **new** repo (not part of a parent repo). You know the exact GitHub repo name (e.g. `MY_NEW_REPO`).

### Steps for Cursor / AI

1. **Add `.gitignore`** in the project root if missing (e.g. `venv/`, `__pycache__/`, `output/`, `*.csv`, `*.parquet`, `.env`, `.DS_Store`).

2. **Initialize Git and first commit**
   - From project root: `git init`
   - `git add .`
   - `git commit -m "Initial commit"`
   - `git branch -M main`

3. **Create the repo on GitHub**
   - **Option 1 (manual):** User goes to https://github.com/new → Repository name = **exact name** (e.g. `MY_NEW_REPO`) → leave "Add a README" **unchecked** → Create repository.
   - **Option 2 (terminal):** If user has a GitHub Personal Access Token (repo scope), run from project root:
     ```bash
     export GITHUB_TOKEN=their_token
     curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" https://api.github.com/user/repos -d '{"name":"MY_NEW_REPO","private":false}'
     ```
     Replace `MY_NEW_REPO` with the actual repo name. User should not paste token in chat; they run the command locally.

4. **Remote and push**
   - **SSH (preferred, no token after setup):**
     - Ensure user has SSH key added to their GitHub account (https://github.com/settings/keys).
     - `git remote add origin git@github.com:USERNAME/REPO_NAME.git`
     - Use the **exact** repo name as on GitHub (including any hyphen or suffix, e.g. `SPY_INTRADAY_TRENDOSC-`).
   - **HTTPS:** `git remote add origin https://github.com/USERNAME/REPO_NAME.git`
   - Then: `git push -u origin main`

5. **If "Repository not found"**
   - Confirm repo exists: open `https://github.com/USERNAME/REPO_NAME` in browser (exact name, including hyphens).
   - If repo name on GitHub has a trailing hyphen or different spelling, run: `git remote set-url origin git@github.com:USERNAME/EXACT_REPO_NAME.git`

6. **If remote already has a first commit (e.g. README)**
   - `git pull origin main --rebase` then resolve any README conflict (keep local content, remove conflict markers), `git add README.md`, `GIT_EDITOR=true git rebase --continue`, then `git push -u origin main`.

7. **Cursor workspace:** To have Source Control show this repo (not a parent), open the **project folder** as the workspace (File → Open Folder → select this project), not the parent directory.

---

## B. Create a new branch in an existing repository

**Prerequisites:** You're inside a Git repo (e.g. `spy_5min_to_w_ema`). You want a new branch for a feature or experiment.

### Steps for Cursor / AI

1. **From repo root, create and switch to the new branch**
   - `git checkout -b BRANCH_NAME`
   - Examples: `feature/atr-exit-v2`, `experiment/15min-timeframe`, `fix/entry-window`

2. **Optional: push the new branch to GitHub**
   - `git push -u origin BRANCH_NAME`
   - After that, "Publish branch" in Cursor will work for this branch.

3. **Reminder:** Commits on this branch stay on this branch until you merge (e.g. via GitHub PR or `git checkout main && git merge BRANCH_NAME`).

---

## Quick reference: repo name and remote

| Item | Example |
|------|--------|
| GitHub username | `MAXIBOI1` |
| Repo name (exact as on GitHub) | `SPY_INTRADAY_TRENDOSC-` |
| SSH remote | `git@github.com:MAXIBOI1/SPY_INTRADAY_TRENDOSC-.git` |
| HTTPS remote | `https://github.com/MAXIBOI1/SPY_INTRADAY_TRENDOSC-.git` |

Always match the repo name **exactly** (including `-` or other characters) as shown on the GitHub repo URL.

---

## Prompt to paste into Cursor for a new repo

Copy and paste the block below when you want to create a **new** repository. Replace the placeholders.

```
I want to create a new GitHub repository from this project folder.

- GitHub username: [YOUR_GITHUB_USERNAME]
- Exact repository name on GitHub: [EXACT_REPO_NAME]
- This folder is the project root and should be the only root (not inside another repo I want to keep).

Follow the process in docs/GIT_NEW_REPO_AND_BRANCH_PROCESS.md section A (new GitHub repo). Add a .gitignore if missing, init Git, create first commit on main, then add remote and push. If the repo already exists on GitHub with a different spelling (e.g. trailing hyphen), use the exact name for the remote. If GitHub already has an initial commit, rebase and resolve README conflict then push.
```

---

## Prompt to paste into Cursor for a new branch

Copy and paste the block below when you want a **new branch** in the current repo.

```
I want to create a new branch in this repository.

- Branch name: [e.g. feature/my-feature or experiment/test]

Follow the process in docs/GIT_NEW_REPO_AND_BRANCH_PROCESS.md section B: create the branch and optionally push it to origin.
```

---

*Last updated for SPY_INTRADAY_TRENDOSC workflow. Adjust USERNAME/REPO_NAME for other projects.*
