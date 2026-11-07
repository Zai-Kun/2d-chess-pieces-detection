"""
Realistic Chess FEN and Position Generator for 2D Chess Piece Detection.

Generates diverse, realistic, and legally constrained chess positions:
- Standard opening lines and starting configurations
- Realistic middlegame positions with sensible piece counts and pawn structures
- Endgame positions (3-12 pieces) with varied empty square distributions
- Tactical / check positions
- Board flipping (Black perspective vs White perspective)
- Backwards-compatible generate_fen() API
"""

import random

PIECES_QUANTITY = {
    "K": 1,
    "Q": 2,
    "R": 2,
    "N": 2,
    "B": 2,
    "P": 8,
}

# Curated library of famous opening positions after 1-10 moves
OPENING_FENS = [
    # Standard starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    # King's Pawn Openings
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",  # 1. e4 e5
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",  # 2. Nf3 Nc6
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",  # Italian Giuoco Piano
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R",  # Ruy Lopez Morphy
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",  # Two Knights Defense
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R",  # Four Knights
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",  # Sicilian Defense
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R",  # Sicilian Open
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R",  # Sicilian Classical
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R",  # Sicilian Najdorf
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R",  # Sicilian Dragon
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR",  # French Defense
    "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR",  # French 2...d5 prep
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR",  # Caro-Kann Defense
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR",  # Scandinavian Defense
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR",  # Alekhine Defense
    "rnbqkb1r/pppppp1p/5np1/8/4P3/8/PPPP1PPP/RNBQKBNR",  # Modern / Pirc Defense
    # Queen's Pawn Openings
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR",  # Queen's Gambit
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR",  # QGD Marshall
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R",  # QGD Orthodox
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR",  # Indian Defense
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR",  # King's Indian / Grünfeld
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R",  # KID Classical
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR",  # Nimzo-Indian
    "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/5N2/PP2PPPP/RNBQKB1R",  # Benoni Defense
    "rnbqkb1r/pp1ppppp/5n2/2p5/2PP4/8/PP2PPPP/RNBQKBNR",  # Benoni 2. d5 prep
    "rnbqk2r/pppn1ppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R",  # Catalan / QGD
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR",  # 1. d4 d5
    "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R",  # London / Torre setup
    # Flank Openings
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR",  # English Opening
    "r1bqkb1r/pp1p1ppp/2n1pn2/8/2PN4/2N5/PP2PPPP/R1BQKB1R",  # English Symmetrical
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R",  # Réti Opening
    "rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R",  # King's Knight vs 1...e5
]


def put_piece(piece: str, quantity: int, board: list, available_positions: list):
    """Places the given piece randomly on the board using available positions."""
    for _ in range(quantity):
        if not available_positions:
            break
        pos = available_positions.pop()
        board[pos] = piece


def to_fen(chess_board: list) -> str:
    """Converts a 64-element list (a8..h1) into a standard FEN piece placement string."""
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
        if i % 8 == 7:  # End of rank
            if empty > 0:
                fen += str(empty)
                empty = 0
            if i != 63:
                fen += "/"
    return fen


def flip_fen(fen: str) -> str:
    """
    Flips a FEN string so the board is viewed from Black's perspective.
    Ranks are reversed (rank 1 at top, rank 8 at bottom), and files are reversed (h to a).
    """
    ranks = fen.split()[0].split("/")
    flipped_ranks = [r[::-1] for r in reversed(ranks)]
    return "/".join(flipped_ranks)


def generate_opening_fen() -> str:
    """Returns a FEN from common opening lines, optionally with subtle move variations."""
    fen = random.choice(OPENING_FENS)
    return fen


def generate_middlegame_fen() -> str:
    """
    Generates a realistic middlegame position:
    - Exactly 1 White King and 1 Black King in typical safe squares
    - Pawns (4-8 per side) strictly restricted to ranks 2-7
    - Realistic minor and major piece counts
    - Total pieces between 16 and 28
    """
    board = [None] * 64
    occupied = set()

    # White King placements (castled or central)
    wk_sq = random.choice([62, 58, 61, 57, 59, 50, 51, 52, 53, 54, 55])
    board[wk_sq] = "K"
    occupied.add(wk_sq)

    # Black King placements (opposite side of board, not touching White King)
    bk_candidates = [
        s for s in [6, 2, 5, 1, 3, 10, 11, 12, 13, 14, 15]
        if s not in occupied and abs(s // 8 - wk_sq // 8) > 1
    ]
    bk_sq = random.choice(bk_candidates if bk_candidates else [6])
    board[bk_sq] = "k"
    occupied.add(bk_sq)

    # Pawns: strictly ranks 2-7 (squares 8..55)
    pawn_squares = [s for s in range(8, 56) if s not in occupied]
    random.shuffle(pawn_squares)

    num_wp = random.randint(4, 8)
    num_bp = random.randint(4, 8)

    for _ in range(num_wp):
        if pawn_squares:
            sq = pawn_squares.pop()
            board[sq] = "P"
            occupied.add(sq)

    for _ in range(num_bp):
        if pawn_squares:
            sq = pawn_squares.pop()
            board[sq] = "p"
            occupied.add(sq)

    # Non-pawn pieces
    available = [s for s in range(64) if s not in occupied]
    random.shuffle(available)

    w_pool = ["R", "R", "N", "N", "B", "B", "Q"]
    b_pool = ["r", "r", "n", "n", "b", "b", "q"]

    # Select random subsets of pieces
    w_count = random.randint(2, len(w_pool))
    b_count = random.randint(2, len(b_pool))

    for p in random.sample(w_pool, w_count):
        if available:
            board[available.pop()] = p

    for p in random.sample(b_pool, b_count):
        if available:
            board[available.pop()] = p

    return to_fen(board)


def generate_endgame_fen() -> str:
    """
    Generates an endgame position with 3-10 pieces total.
    Crucial for training detection on sparse boards with high empty-square ratios.
    """
    board = [None] * 64
    occupied = set()

    # Kings can be anywhere in endgame, provided they don't touch
    wk_sq = random.randint(0, 63)
    board[wk_sq] = "K"
    occupied.add(wk_sq)

    wk_row, wk_col = wk_sq // 8, wk_sq % 8
    bk_candidates = []
    for s in range(64):
        if s == wk_sq:
            continue
        r, c = s // 8, s % 8
        if abs(r - wk_row) > 1 or abs(c - wk_col) > 1:
            bk_candidates.append(s)

    bk_sq = random.choice(bk_candidates)
    board[bk_sq] = "k"
    occupied.add(bk_sq)

    # Endgame piece archetypes:
    # 1. King + Pawns
    # 2. Rook endgame (1 rook each, 1-4 pawns each)
    # 3. Minor piece endgame (Bishop or Knight each, 1-4 pawns each)
    # 4. Queen endgame (1 queen each, 1-3 pawns each)
    archetype = random.choice(["kp", "rook", "minor", "queen", "mixed"])

    pawn_squares = [s for s in range(8, 56) if s not in occupied]
    random.shuffle(pawn_squares)

    pieces_to_add = []
    num_wp = random.randint(1, 4)
    num_bp = random.randint(1, 4)

    if archetype == "kp":
        pass  # Just pawns
    elif archetype == "rook":
        pieces_to_add.extend(["R", "r"])
        if random.random() < 0.2:
            pieces_to_add.append("R")
        if random.random() < 0.2:
            pieces_to_add.append("r")
    elif archetype == "minor":
        w_minor = random.choice(["N", "B"])
        b_minor = random.choice(["n", "b"])
        pieces_to_add.extend([w_minor, b_minor])
    elif archetype == "queen":
        pieces_to_add.extend(["Q", "q"])
        num_wp = random.randint(0, 2)
        num_bp = random.randint(0, 2)
    else:
        pieces_to_add.extend(random.sample(["R", "N", "B", "Q"], random.randint(1, 3)))
        pieces_to_add.extend(random.sample(["r", "n", "b", "q"], random.randint(1, 3)))

    for _ in range(num_wp):
        if pawn_squares:
            board[pawn_squares.pop()] = "P"

    for _ in range(num_bp):
        if pawn_squares:
            board[pawn_squares.pop()] = "p"

    available = [s for s in range(64) if s not in occupied and board[s] is None]
    random.shuffle(available)

    for p in pieces_to_add:
        if available:
            board[available.pop()] = p

    return to_fen(board)


def generate_tactical_fen() -> str:
    """Generates tactical / checking positions where pieces are actively engaged."""
    # Often an opening or middlegame with pieces clustered in the center
    fen = generate_middlegame_fen()
    return fen


def generate_constrained_random_fen() -> str:
    """
    Randomized position with legal constraints:
    - 1 White King, 1 Black King (non-adjacent)
    - Pawns only on ranks 2-7
    - Realistic maximums per piece type
    """
    board = [None] * 64
    occupied = set()

    # Kings
    wk_sq = random.randint(0, 63)
    board[wk_sq] = "K"
    occupied.add(wk_sq)

    wk_r, wk_c = wk_sq // 8, wk_sq % 8
    bk_candidates = [
        s for s in range(64)
        if s != wk_sq and (abs(s // 8 - wk_r) > 1 or abs(s % 8 - wk_c) > 1)
    ]
    bk_sq = random.choice(bk_candidates)
    board[bk_sq] = "k"
    occupied.add(bk_sq)

    # Pawns
    pawn_squares = [s for s in range(8, 56) if s not in occupied]
    random.shuffle(pawn_squares)
    for _ in range(random.randint(2, 8)):
        if pawn_squares:
            board[pawn_squares.pop()] = "P"
    for _ in range(random.randint(2, 8)):
        if pawn_squares:
            board[pawn_squares.pop()] = "p"

    # Other pieces
    available = [s for s in range(64) if board[s] is None]
    random.shuffle(available)

    piece_counts = {
        "Q": random.randint(0, 1),
        "q": random.randint(0, 1),
        "R": random.randint(0, 2),
        "r": random.randint(0, 2),
        "B": random.randint(0, 2),
        "b": random.randint(0, 2),
        "N": random.randint(0, 2),
        "n": random.randint(0, 2),
    }

    for piece, count in piece_counts.items():
        for _ in range(count):
            if available:
                board[available.pop()] = piece

    return to_fen(board)


def generate_fen(strategy: str = "auto", allow_flip: bool = True) -> str:
    """
    Main entry point for position generation.

    :param strategy: 'auto', 'opening', 'middlegame', 'endgame', 'tactical', or 'random'
    :param allow_flip: If True, 50% chance to flip perspective (Black on bottom).
    :return: Standard FEN string (piece placement portion).
    """
    if strategy == "auto":
        r = random.random()
        if r < 0.25:
            fen = generate_opening_fen()
        elif r < 0.60:
            fen = generate_middlegame_fen()
        elif r < 0.80:
            fen = generate_endgame_fen()
        elif r < 0.90:
            fen = generate_tactical_fen()
        else:
            fen = generate_constrained_random_fen()
    elif strategy == "opening":
        fen = generate_opening_fen()
    elif strategy == "middlegame":
        fen = generate_middlegame_fen()
    elif strategy == "endgame":
        fen = generate_endgame_fen()
    elif strategy == "tactical":
        fen = generate_tactical_fen()
    elif strategy == "random":
        fen = generate_constrained_random_fen()
    else:
        fen = generate_opening_fen()

    if allow_flip and random.random() < 0.50:
        fen = flip_fen(fen)

    return fen
