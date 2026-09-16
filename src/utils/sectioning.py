def get_sections_to_do(game_dir, prefix, sections_arg):
    if sections_arg is None:
        # default to all
        return game_dir.glob(prefix+"*")
    else:
        return [game_dir / (prefix+s) for s in sections_arg]

def get_section_names_to_do(game_dir, prefix, sections_arg):
    if sections_arg is None:
        # default to all
        return [p.name for p in game_dir.glob(prefix+"*")]
    else:
        return [prefix+s for s in sections_arg]
