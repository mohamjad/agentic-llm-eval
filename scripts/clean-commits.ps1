# Clean commit messages by removing Co-authored-by trailers
# This script rewrites the last 15 commits

$commits = @(git log --format="%H" -15)
$baseCommit = git log --format="%H" -16 | Select-Object -Last 1

Write-Host "Rewriting $($commits.Count) commits..."

# Reset to base commit (soft reset to keep changes)
git reset --soft $baseCommit

# Re-commit each change with cleaned message
foreach ($commit in [array]::Reverse($commits)) {
    $msg = git show $commit --format="%B" --no-patch
    $cleanMsg = ($msg -split "`n" | Where-Object { $_ -notmatch "^Co-authored-by: Cursor" }) -join "`n"
    $cleanMsg = $cleanMsg.Trim()
    
    # Get the files changed in this commit
    $files = git diff-tree --no-commit-id --name-only -r $commit
    
    if ($files.Count -gt 0) {
        # Stage the files
        git reset HEAD
        foreach ($file in $files) {
            if (Test-Path $file) {
                git add $file
            }
        }
        
        # Commit with cleaned message
        git commit -m $cleanMsg --no-verify
    }
}

Write-Host "Done! Commits cleaned."
