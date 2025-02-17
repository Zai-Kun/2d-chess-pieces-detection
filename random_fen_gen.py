import random

PIECES_QUANTITY = {
    "K": 1,
    "Q": 1,
    "R": 2,
    "N": 2,
    "B": 2,
    "P": 8,
}

def put_piece(piece: str, quantity: int, board: list, available_positions: list):
    """Places the given piece randomly on the board using available positions."""
    for _ in range(quantity):
        if not available_positions:
            break
        pos = available_positions.pop()  # Get and remove a random position
        board[pos] = piece

def to_fen(chess_board: list):
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
        if i % 8 == 7:  # End of a row
            if empty > 0:
                fen += str(empty)
                empty = 0
            if i != 63:
                fen += '/'
    return fen

def generate_fen():
    """Generates a random chess FEN position efficiently."""
    chess_board = [None] * 64  # Use a fixed list to avoid reallocation
    available_positions = list(range(64))  # Precompute positions
    random.shuffle(available_positions)  # Shuffle once to avoid multiple `randint` calls

    for i in range(2):  # Black and white pieces
        for piece, quantity in PIECES_QUANTITY.items():
            piece_symbol = piece if i == 0 else piece.lower()
            put_piece(piece_symbol, 1 if piece == "K" else random.randint(0, quantity), chess_board, available_positions)

    return to_fen(chess_board)
