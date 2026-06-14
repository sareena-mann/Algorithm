# Camelot wheel: (spotify_key 0-11, mode 0=minor 1=major) -> (number, letter)
CAMELOT = {
    (0, 1): (8, 'B'),   (1, 1): (3, 'B'),   (2, 1): (10, 'B'),
    (3, 1): (5, 'B'),   (4, 1): (12, 'B'),  (5, 1): (7, 'B'),
    (6, 1): (2, 'B'),   (7, 1): (9, 'B'),   (8, 1): (4, 'B'),
    (9, 1): (11, 'B'),  (10, 1): (6, 'B'),  (11, 1): (1, 'B'),
    (0, 0): (5, 'A'),   (1, 0): (12, 'A'),  (2, 0): (7, 'A'),
    (3, 0): (2, 'A'),   (4, 0): (9, 'A'),   (5, 0): (4, 'A'),
    (6, 0): (11, 'A'),  (7, 0): (6, 'A'),   (8, 0): (1, 'A'),
    (9, 0): (8, 'A'),   (10, 0): (3, 'A'),  (11, 0): (10, 'A'),
}

# Plain-English key name (from GetSongBPM) -> Camelot string
KEY_TO_CAMELOT = {
    "C Major": "8B",   "C# Major": "3B",  "Db Major": "3B",
    "D Major": "10B",  "D# Major": "5B",  "Eb Major": "5B",
    "E Major": "12B",  "F Major": "7B",   "F# Major": "2B",  "Gb Major": "2B",
    "G Major": "9B",   "G# Major": "4B",  "Ab Major": "4B",
    "A Major": "11B",  "A# Major": "6B",  "Bb Major": "6B",  "B Major": "1B",
    "C Minor": "5A",   "C# Minor": "12A", "Db Minor": "12A",
    "D Minor": "7A",   "D# Minor": "2A",  "Eb Minor": "2A",
    "E Minor": "9A",   "F Minor": "4A",   "F# Minor": "11A", "Gb Minor": "11A",
    "G Minor": "6A",   "G# Minor": "1A",  "Ab Minor": "1A",
    "A Minor": "8A",   "A# Minor": "3A",  "Bb Minor": "3A",  "B Minor": "10A",
}


def get_camelot(key: int, mode: int) -> str:
    entry = CAMELOT.get((key, mode))
    return f"{entry[0]}{entry[1]}" if entry else "?"


def camelot_compat(c1: str, c2: str) -> str:
    """Returns 'perfect', 'good', or 'clash'."""
    if not c1 or not c2 or "?" in (c1, c2):
        return "unknown"
    try:
        num1, let1 = int(c1[:-1]), c1[-1]
        num2, let2 = int(c2[:-1]), c2[-1]
    except (ValueError, IndexError):
        return "unknown"
    if num1 == num2:
        return "perfect"
    if let1 == let2:
        diff = abs(num1 - num2)
        if diff == 1 or diff == 11:
            return "perfect"
        if diff == 2 or diff == 10:
            return "good"
    return "clash"
