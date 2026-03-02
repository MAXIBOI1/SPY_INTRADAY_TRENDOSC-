#!/usr/bin/env bash
# Creates GitHub repo SPY_INTRADAY_TRENDOSC (run once).
# Usage: export GITHUB_TOKEN=your_token; ./scripts/create_github_repo.sh
set -e
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Set GITHUB_TOKEN first: export GITHUB_TOKEN=ghp_xxxx"
  exit 1
fi
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{"name":"SPY_INTRADAY_TRENDOSC","private":false}'
echo ""
echo "Repo created. Run: git push -u origin main"
