#!/bin/bash
# Script to remove Co-authored-by trailers from git history
# Usage: Run this script, then force push: git push --force origin master

git filter-branch --force --msg-filter '
    sed "/^Co-authored-by: Cursor/d"
' -- --all

# Clean up backup refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

echo "Done! Co-authored-by trailers removed."
echo "Review changes, then force push: git push --force origin master"
