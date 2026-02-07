# Remove Co-authored-by trailers from recent commits
$commits = git log --format="%H" -15
$baseCommit = git log --format="%H" -16 | Select-Object -Last 1

foreach ($commit in $commits) {
    $msg = git show $commit --format="%B" --no-patch
    $cleanMsg = $msg -replace "(?m)^Co-authored-by: Cursor.*\r?\n", ""
    
    if ($msg -ne $cleanMsg) {
        git commit --amend -m $cleanMsg --no-verify
    }
}
