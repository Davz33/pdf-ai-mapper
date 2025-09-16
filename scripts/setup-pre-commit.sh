#!/bin/bash
# Setup script to install trufflehog pre-commit hook

echo "🔧 Setting up trufflehog pre-commit hook..."

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy the pre-commit hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit

echo "✅ Pre-commit hook installed successfully!"
echo "Trufflehog will now run before every commit to scan for secrets."
