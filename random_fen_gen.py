import os
import random
from typing import List, Optional

CHESS_COM_FENS_FILE = "assets/chess_com_fens.txt"

# Default fallback list of realistic FEN positions from famous games / chess.com archives
# used if assets/chess_com_fens.txt has not been fully built yet.
FALLBACK_CHESS_COM_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQKB1R",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR",
    "r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1",
    "rnbq1rk1/ppp1ppbp/5np1/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",
    "r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/8/PPP2PPP/RNBQKB1R",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R",
    "rnbq1rk1/ppp2ppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R",
    "r1bqkbnr/pp3ppp/2n5/2pp4/3P4/5N2/PPP1BPPP/RNBQK2R",
    "8/8/4k3/8/8/4K3/8/8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1",
]

_cached_chess_com_fens: Optional[List[str]] = None

def get_chess_com_fens_list() -> List[str]:
    """Loads cached Chess.com FEN positions from file or fallback list."""
    global _cached_chess_com_fens
    if _cached_chess_com_fens is not None and len(_cached_chess_com_fens) > 0:
        return _cached_chess_com_fens

    fens = []
    if os.path.exists(CHESS_COM_FENS_FILE):
        with open(CHESS_COM_FENS_FILE, "r") as f:
            for line in f:
                fen = line.strip().split()[0]
                if fen and fen.count("/") == 7:
                    fens.append(fen)

    if not fens:
        fens = FALLBACK_CHESS_COM_FENS

    _cached_chess_com_fens = fens
    return fens

def board_to_fen(chess_board: list) -> str:
    """Converts a 64-element list representation of a board into a FEN board placement string."""
    fen = ""
    empty = 0
    for i, piece in enumerate(chess_board):
        if piece is None:
            empty += 1
        else:
            if empty > 0:
                fen += str(empty)
                empty = 0
            fen += piece
        if i % 8 == 7:  # End of a rank
            if empty > 0:
                fen += str(empty)
                empty = 0
            if i != 63:
                fen += "/"
    return fen

def get_random_chess_com_fen() -> str:
    """70% Source: Selects a random FEN from Chess.com's open database pool."""
    fens = get_chess_com_fens_list()
    return random.choice(fens)

def generate_custom_realistic_fen() -> str:
    """
    20% Source: Generates a realistic custom chess position.
    Follows standard chess piece counts and pawn rank constraints.
    """
    board = [None] * 64
    available_indices = list(range(64))
    random.shuffle(available_indices)

    # Place Kings (1 White King, 1 Black King)
    w_king_pos = available_indices.pop()
    board[w_king_pos] = "K"
    
    b_king_pos = available_indices.pop()
    board[b_king_pos] = "k"

    # Define realistic piece distribution counts per color
    # Pawns: 0-8, Rooks: 0-2, Knights: 0-2, Bishops: 0-2, Queens: 0-2
    piece_pools = {
        "P": random.randint(0, 8),
        "R": random.randint(0, 2),
        "N": random.randint(0, 2),
        "B": random.randint(0, 2),
        "Q": random.randint(0, 2),
    }

    # White pieces
    for p_type, count in piece_pools.items():
        for _ in range(count):
            if not available_indices:
                break
            if p_type == "P":
                # Pawns cannot be on 1st rank (indices 56-63) or 8th rank (indices 0-7)
                pawn_indices = [idx for idx in available_indices if 8 <= idx <= 55]
                if not pawn_indices:
                    continue
                pos = random.choice(pawn_indices)
                available_indices.remove(pos)
            else:
                pos = available_indices.pop()
            board[pos] = p_type

    # Black pieces
    b_piece_pools = {
        "p": random.randint(0, 8),
        "r": random.randint(0, 2),
        "n": random.randint(0, 2),
        "b": random.randint(0, 2),
        "q": random.randint(0, 2),
    }

    for p_type, count in b_piece_pools.items():
        for _ in range(count):
            if not available_indices:
                break
            if p_type == "p":
                pawn_indices = [idx for idx in available_indices if 8 <= idx <= 55]
                if not pawn_indices:
                    continue
                pos = random.choice(pawn_indices)
                available_indices.remove(pos)
            else:
                pos = available_indices.pop()
            board[pos] = p_type

    return board_to_fen(board)

def generate_bogus_fen() -> str:
    """
    10% Source: Generates a complete bogus / chaotic position.
    Can feature extreme piece counts (e.g. 8 queens, 5 kings), wild piece scatter,
    or extremely sparse/dense boards to make YOLO robust against illegal/unusual patterns.
    """
    board = [None] * 64
    all_pieces = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]
    
    # Random piece count anywhere from 1 to 50 pieces
    num_pieces = random.randint(1, 50)
    positions = random.sample(range(64), num_pieces)

    for pos in positions:
        # Completely unconstrained piece selection
        board[pos] = random.choice(all_pieces)

    return board_to_fen(board)

def generate_fen(ratio=(0.70, 0.20, 0.10)) -> str:
    """
    Generates a FEN position based on the required distribution:
      - 70% Chess.com Open Database
      - 20% Custom Realistic Generator
      - 10% Complete Bogus Generator
    """
    r = random.random()
    p_chess_com, p_custom, p_bogus = ratio
    
    if r < p_chess_com:
        return get_random_chess_com_fen()
    elif r < p_chess_com + p_custom:
        return generate_custom_realistic_fen()
    else:
        return generate_bogus_fen()

if __name__ == "__main__":
    print("Testing FEN Generator Distribution (10 samples):")
    for idx in range(10):
        fen = generate_fen()
        print(f"Sample {idx+1:2d}: {fen}")
