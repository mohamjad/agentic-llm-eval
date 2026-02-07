#!/bin/bash
# Remove Co-authored-by trailers from commit messages
git filter-branch --msg-filter '
    sed "/^Co-authored-by: Cursor/d"
' -- --all
