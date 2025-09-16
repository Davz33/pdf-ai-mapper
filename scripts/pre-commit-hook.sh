#!/bin/bash
# Pre-commit hook to run trufflehog security scan

echo "🔍 Running trufflehog security scan..."

# Get the current repository path
REPO_PATH=$(git rev-parse --show-toplevel)

# Run trufflehog on the repository, excluding GitHub detector (false positives in lock files)
trufflehog git file://"$REPO_PATH" --no-verification --exclude-detectors="Github"

# Capture the exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Trufflehog scan passed - no secrets detected"
    exit 0
else
    echo "❌ Trufflehog scan failed - potential secrets detected"
    echo "Please review the output above and remove any secrets before committing"
    exit 1
fi
