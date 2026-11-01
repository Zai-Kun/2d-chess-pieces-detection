
#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <ASSET_ID> [headers_file]"
    exit 1
fi

ID="$1"
DIR="pieces/$ID"
HEADER_FILE="${2:-headers.txt}"
BASE_URL="https://assets-themes.chess.com/image/$ID/300"

pieces=(
    wp wn wb wr wq wk
    bp bn bb br bq bk
)

curl_args=(
    --fail
    --location
    --progress-bar
    --compressed
    --http2
)

# Load headers from file
if [[ -f "$HEADER_FILE" ]]; then
    echo "Loading headers from $HEADER_FILE..."

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Handle Windows line endings
        line="${line%$'\r'}"

        # Ignore request lines, empty lines and invalid lines
        [[ "$line" == *:* ]] || continue

        name="${line%%:*}"
        value="${line#*:}"

        # Trim whitespace
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        [[ -n "$name" ]] || continue

        # Let curl handle transport-specific headers
        case "${name,,}" in
            host|connection|proxy-connection|keep-alive|\
            transfer-encoding|upgrade|te|accept-encoding)
                continue
                ;;
        esac

        curl_args+=(-H "$name: $value")
    done < "$HEADER_FILE"
elif [[ $# -eq 2 ]]; then
    echo "Error: Header file '$HEADER_FILE' not found."
    exit 1
else
    echo "No headers.txt found. Using curl defaults."
fi

mkdir -p "$DIR"

for piece in "${pieces[@]}"; do
    echo "Downloading $piece.png..."

    curl "${curl_args[@]}" \
        "$BASE_URL/$piece.png" \
        -o "$DIR/$piece.png"
done

echo
echo "Done! Saved theme to: $DIR/"
