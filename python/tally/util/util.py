def split_and_strip(_str, splitter=" ", max_count=-1, rsplit=False):
    if not rsplit:
        parts = _str.split(splitter, max_count)
    else:
        parts = _str.rsplit(splitter, max_count)
    new_parts = []
    for part in parts:
        new_part = part.strip()
        if new_part:
            new_parts.append(new_part)
    
    return new_parts


def remove_keywords(_str, ignore_keywords):

    for keyword in ignore_keywords:
        if keyword in _str:

            idx = _str.index(keyword)

            # need to ensure keyword is only by itself, not within a name
            if idx > 0 and _str[idx - 1] != " ":
                continue
            if idx + len(keyword) < len(_str) and _str[idx + len(keyword)] != " ":
                continue

            _str = _str.replace(keyword, " ")
    
    return _str.strip()