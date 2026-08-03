import json
import os
import re
import urllib.request
from typing import List, Set

CHESS_COM_FENS_FILE = "assets/chess_com_fens.txt"

# Featured top players / GMs on Chess.com
TARGET_PLAYERS = [
    "hikaru",
    "magnuscarlsen",
    "danielnaroditsky",
    "gothamchess",
    "fabianocaruana",
    "firouzja2003",
    "nihalsarin",
    "praggchess",
    "viswanathananand",
    "chichess",
]

def extract_fen_board_part(fen: str) -> str:
    """Extracts only the piece placement part of a FEN string (before the first space)."""
    return fen.strip().split()[0]

def parse_pgn_fens(pgn_text: str) -> List[str]:
    """
    Parses a PGN text and simulates/extracts board positions.
    Extracts [FEN "..."] headers or reconstructs piece placements.
    """
    fens = []
    # Check for explicit FEN tags
    fen_matches = re.findall(r'\[FEN\s+"([^"]+)"\]', pgn_text)
    for f in fen_matches:
        board_part = extract_fen_board_part(f)
        if board_part and "/" in board_part:
            fens.append(board_part)
    return fens

def fetch_games_for_player(username: str, max_months: int = 3) -> Set[str]:
    """Fetches recent game archives for a given player from Chess.com public API."""
    extracted_fens = set()
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    
    try:
        req = urllib.request.Request(
            archives_url,
            headers={"User-Agent": "ChessBoardDatasetGenerator/1.0 (contact: user@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            archive_urls = data.get("archives", [])
    except Exception as e:
        print(f"Failed to fetch archives for {username}: {e}")
        return extracted_fens

    # Take recent months
    recent_archives = archive_urls[-max_months:]
    for url in recent_archives:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ChessBoardDatasetGenerator/1.0 (contact: user@example.com)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                month_data = json.loads(resp.read().decode("utf-8"))
                games = month_data.get("games", [])
                for game in games:
                    # Final FEN if present
                    if "fen" in game:
                        board_part = extract_fen_board_part(game["fen"])
                        if "/" in board_part and board_part.count("/") == 7:
                            extracted_fens.add(board_part)
                    
                    # PGN FENs or initial setup
                    if "pgn" in game:
                        pgn_fens = parse_pgn_fens(game["pgn"])
                        for pf in pgn_fens:
                            if pf.count("/") == 7:
                                extracted_fens.add(pf)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue

    return extracted_fens

def build_chess_com_fen_database(limit: int = 15000) -> List[str]:
    """Builds and caches a dataset of real Chess.com FEN positions."""
    os.makedirs(os.path.dirname(CHESS_COM_FENS_FILE), exist_ok=True)
    all_fens = set()

    # Load existing if available
    if os.path.exists(CHESS_COM_FENS_FILE):
        with open(CHESS_COM_FENS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and line.count("/") == 7:
                    all_fens.add(line)

    print(f"Currently cached Chess.com FENs: {len(all_fens)}")
    
    if len(all_fens) < limit:
        print("Fetching fresh Chess.com game FENs from public API...")
        for player in TARGET_PLAYERS:
            print(f"  Fetching for player '{player}'...")
            player_fens = fetch_games_for_player(player, max_months=2)
            all_fens.update(player_fens)
            print(f"  Total unique FENs so far: {len(all_fens)}")
            if len(all_fens) >= limit:
                break

    fen_list = sorted(list(all_fens))
    with open(CHESS_COM_FENS_FILE, "w") as f:
        for fen in fen_list:
            f.write(f"{fen}\n")

    print(f"Successfully saved {len(fen_list)} Chess.com FENs to {CHESS_COM_FENS_FILE}")
    return fen_list

if __name__ == "__main__":
    build_chess_com_fen_database()
