import munch

def timestamp_add(a, b):
    tot = munch.Munch(
        secs=a.secs+b.secs,
        nanos=a.nanos+b.nanos,
    )
    while tot.nanos >= 1_000_000_000:
        tot.secs += 1
        tot.nanos -= 1_000_000_000
    return tot

def timestamp_diff(a, b):
    diff = munch.Munch(
        secs=a.secs-b.secs,
        nanos=a.nanos-b.nanos,
    )
    while diff.nanos < 0:
        diff.secs -= 1
        diff.nanos += 1_000_000_000
    return diff

# milliseconds
def timestamp_to_float(t):
    return (t.secs + (t.nanos / 1e9)) * 1000

def filter_dict_recursive(d, keys_to_keep):
    result = {}
    for key, value in d.items():
        if key in keys_to_keep:
            if isinstance(value, dict):
                result[key] = filter_dict_recursive(value, keys_to_keep[key])
            elif isinstance(value, list):
                result[key] = [filter_dict_recursive(elem, keys_to_keep[key]) for elem in value]
            else:
                result[key] = value
    return result

def flatten_dict_recursive(d, sep="_"):
    items = {}

    def _flatten(val, key):
        if isinstance(val, dict):
            for k, v in val.items():
                new_key = f"{key}{sep}{k}" if key else str(k)
                _flatten(v, new_key)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if key.endswith("players"):
                    i += 1  # work with jersey numbers
                new_key = f"{key}{sep}{i}" if key else str(i)
                _flatten(item, new_key)
        else:
            items[key] = val

    _flatten(d, "")
    return items
