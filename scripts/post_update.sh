#!/bin/bash
# Post an update as a GitHub Discussion in the repo's Announcements category.
#
# Usage:
#   scripts/post_update.sh "Title here" path/to/body.md
#   scripts/post_update.sh "Title here" -          # read body from stdin
#
# Requires: gh CLI authenticated.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <title> <body.md|->" >&2
  exit 1
fi

TITLE="$1"
BODY_SRC="$2"

REPO_OWNER="yanghangAI"
REPO_NAME="autosapiensiter"
CATEGORY="Announcements"

# Resolve repo + category IDs dynamically so this keeps working if the repo moves.
IDS=$(gh api graphql -f query="
query {
  repository(owner:\"$REPO_OWNER\", name:\"$REPO_NAME\") {
    id
    discussionCategories(first:20) { nodes { id name } }
  }
}")

REPO_ID=$(printf '%s' "$IDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['repository']['id'])")
CAT_ID=$(printf '%s' "$IDS" | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']['repository']['discussionCategories']['nodes']
cat=[c for c in d if c['name']=='$CATEGORY'][0]['id']
print(cat)")

if [ "$BODY_SRC" = "-" ]; then
  BODY_FILE=$(mktemp)
  trap 'rm -f "$BODY_FILE"' EXIT
  cat > "$BODY_FILE"
else
  BODY_FILE="$BODY_SRC"
fi

gh api graphql \
  -F repoId="$REPO_ID" \
  -F catId="$CAT_ID" \
  -F title="$TITLE" \
  -F body=@"$BODY_FILE" \
  -f query='
mutation($repoId:ID!, $catId:ID!, $title:String!, $body:String!) {
  createDiscussion(input:{
    repositoryId:$repoId, categoryId:$catId, title:$title, body:$body
  }) { discussion { url } }
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['createDiscussion']['discussion']['url'])"
