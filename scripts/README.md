# Scripts

Utility scripts for development and maintenance.

## `remove-coauthors.sh`

Removes "Co-authored-by: Cursor" trailers from git commit history.

**Usage:**
```bash
bash scripts/remove-coauthors.sh
git push --force origin master
```

**Warning:** This rewrites git history. Only use if you haven't shared the repo with others, or coordinate with your team first.

## `create_commits.py`

Script for creating realistic commit history. Used for development purposes.

**Note:** This is a development utility and not part of the main package.
