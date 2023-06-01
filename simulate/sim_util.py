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

import itertools

def lcs_lens(xs, ys):
    curr = list(itertools.repeat(0, 1 + len(ys)))
    for x in xs:
        prev = list(curr)
        for i, y in enumerate(ys):
            if x == y:
                curr[i + 1] = prev[i] + 1
            else:
                curr[i + 1] = max(curr[i], prev[i + 1])
    return curr

def longest_common_sequence(xs, ys):
    nx, ny = len(xs), len(ys)
    if nx == 0:
        return []
    elif nx == 1:
        return [xs[0]] if xs[0] in ys else []
    else:
        i = nx // 2
        xb, xe = xs[:i], xs[i:]
        ll_b = lcs_lens(xb, ys)
        ll_e = lcs_lens(xe[::-1], ys[::-1])
        _, k = max((ll_b[j] + ll_e[ny - j], j)
                    for j in range(ny + 1))
        yb, ye = ys[:k], ys[k:]
        return longest_common_sequence(xb, yb) + longest_common_sequence(xe, ye)

if __name__ == "__main__":

    assert ''.join(longest_common_sequence('', '')) == ''
    assert ''.join(longest_common_sequence('a', '')) == ''
    assert ''.join(longest_common_sequence('', 'b')) == ''
    assert ''.join(longest_common_sequence('abc', 'abc')) == 'abc'
    assert ''.join(longest_common_sequence('abcd', 'obce')) == 'bc'
    assert ''.join(longest_common_sequence('abc', 'ab')) == 'ab'
    assert ''.join(longest_common_sequence('abc', 'bc')) == 'bc'
    assert ''.join(longest_common_sequence('abcde', 'zbodf')) == 'bd'
    assert ''.join(longest_common_sequence('aa','aaaa')) == 'aa'
    assert ''.join(longest_common_sequence('GTCGTTCGGAATGCCGTTGCTCTGTAAA',
                        'ACCGGTCGAGTGCGCGGAAGCCGGCCGAA')
                    ) == 'GTCGTCGGAAGCCGGCCGAA'