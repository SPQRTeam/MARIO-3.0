# # this is potentially useful, but currently unused.
# # I'm leaving it commented b/c I just found an inconsistency where sth
# # used this when wanting to use the other, so while this IS available
# # for uncommenting, make sure full absolute paths are what you want!
# def get_sections_to_do(game_dir, prefix, sections_arg):
#     if sections_arg is None:
#         # default to all
#         return game_dir.glob(prefix+"*")
#     else:
#         return [game_dir / (prefix+s) for s in sections_arg]

def get_section_names_to_do(game_dir, prefix, sections_arg):
    if sections_arg is None:
        # default to all
        return [p.name for p in game_dir.glob(prefix+"*")]
    else:
        return [prefix+s for s in sections_arg]
