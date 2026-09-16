import re
from datetime import timedelta

def intn(s):
    if s is None: return 0
    return int(s)

def parse_deltas(s):
    m = re.match(r"[*]? (?:(\d+):)? (\d+): (\d+) ( - (?:(\d+):)? (\d+): (\d+) )?", s, re.VERBOSE)
    if m is None:
        raise ValueError("regex")
    start = timedelta(hours=intn(m[1]), minutes=intn(m[2]), seconds=intn(m[3]))
    if m[4] is None:
        return start, None
    end = timedelta(hours=intn(m[5]), minutes=intn(m[6]), seconds=intn(m[7]))
    return start, end
