"""
Project-wide constants.

We keep Showdown format IDs here so everyone uses the same strings.
"""

# Main Week-1 target: official "Randomized 3v3 Singles" (Showdown) — BSS-style
# factory sets, Flat Rules / VGC timer; team preview, three active Pokémon.
FORMAT_MAIN = "gen9bssfactory"

# Full six in standard singles randbats (global random-battle generator).
# This is *not* human-built OU/Ubers teams; it is a long-horizon **proxy** for
# "all six Pokémon may enter sequentially" vs BSS Factory's three-actives cap.
FORMAT_LONG_HORIZON = "gen9randombattle"

# Optional pilot / report: full **6v6** singles with tier-themed factory sets
# (Smogon Battle Factory). Still auto-generated — closer to "six used in order
# under tier-appropriate sets" than randbats, but still not custom ladder teams.
FORMAT_PILOT_SIX_FACTORY = "gen9battlefactory"

