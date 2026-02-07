# Script to clean commit message (remove Co-authored-by line)
$inputFile = $args[0]
$content = Get-Content $inputFile -Raw
$cleaned = $content -replace "(?m)^Co-authored-by: Cursor.*\r?\n", ""
Set-Content -Path $inputFile -Value $cleaned -NoNewline
