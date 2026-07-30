from std.sys.info import simd_width_of


comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def gram_equal(tokens_a: IPtr, start_a: Int, tokens_b: IPtr, start_b: Int, n: Int) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    while k + W <= n:
        var equal = (
            tokens_a.load[width=W](start_a + k)
            == tokens_b.load[width=W](start_b + k)
        )
        if not equal:
            return False
        k += W
    while k < n:
        if tokens_a[start_a + k] != tokens_b[start_b + k]:
            return False
        k += 1
    return True


def gram_slot(tokens: IPtr, start: Int, n: Int, mask: Int) -> Int:
    var slot = 0
    for k in range(n):
        slot = (slot * 1000003 + Int(tokens[start + k])) & mask
    return slot


@export("mrs_unigram_overlap")
def mrs_unigram_overlap(
    target_addr: Int,
    target_len: Int,
    prediction_addr: Int,
    prediction_len: Int,
    counts_addr: Int,
) abi("C") -> Int:
    if target_len < 0 or prediction_len < 0:
        return -1
    if target_len == 0 or prediction_len == 0:
        return 0
    if target_addr == 0 or prediction_addr == 0 or counts_addr == 0:
        return -1

    var target = IPtr(unsafe_from_address=target_addr)
    var prediction = IPtr(unsafe_from_address=prediction_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    for i in range(target_len):
        counts[Int(target[i])] += 1

    var overlap = 0
    for i in range(prediction_len):
        var token = Int(prediction[i])
        if counts[token] > 0:
            counts[token] -= 1
            overlap += 1
    return overlap


@export("mrs_ngram_overlap")
def mrs_ngram_overlap(
    target_addr: Int,
    target_len: Int,
    prediction_addr: Int,
    prediction_len: Int,
    n: Int,
    starts_addr: Int,
    counts_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if target_len < 0 or prediction_len < 0 or n <= 0:
        return -1
    if target_len < n or prediction_len < n:
        return 0
    if (
        capacity <= 0
        or (capacity & (capacity - 1)) != 0
        or target_addr == 0
        or prediction_addr == 0
        or starts_addr == 0
        or counts_addr == 0
    ):
        return -1

    var target = IPtr(unsafe_from_address=target_addr)
    var prediction = IPtr(unsafe_from_address=prediction_addr)
    var starts = IPtr(unsafe_from_address=starts_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var mask = capacity - 1

    for start in range(target_len - n + 1):
        var slot = gram_slot(target, start, n, mask)
        while starts[slot] != 0:
            var existing = Int(starts[slot]) - 1
            if gram_equal(target, existing, target, start, n):
                break
            slot = (slot + 1) & mask
        if starts[slot] == 0:
            starts[slot] = Int64(start + 1)
        counts[slot] += 1

    var overlap = 0
    for start in range(prediction_len - n + 1):
        var slot = gram_slot(prediction, start, n, mask)
        while starts[slot] != 0:
            var existing = Int(starts[slot]) - 1
            if gram_equal(target, existing, prediction, start, n):
                if counts[slot] > 0:
                    counts[slot] -= 1
                    overlap += 1
                break
            slot = (slot + 1) & mask
    return overlap


@export("mrs_lcs_length")
def mrs_lcs_length(
    ref_addr: Int,
    ref_len: Int,
    candidate_addr: Int,
    candidate_len: Int,
    row_addr: Int,
) abi("C") -> Int:
    if ref_len < 0 or candidate_len < 0:
        return -1
    if ref_len == 0 or candidate_len == 0:
        return 0
    if ref_addr == 0 or candidate_addr == 0 or row_addr == 0:
        return -1

    var ref_tokens = IPtr(unsafe_from_address=ref_addr)
    var candidate = IPtr(unsafe_from_address=candidate_addr)
    var row = IPtr(unsafe_from_address=row_addr)
    for j in range(candidate_len + 1):
        row[j] = 0

    for i in range(1, ref_len + 1):
        var diagonal = Int64(0)
        for j in range(1, candidate_len + 1):
            var above = row[j]
            if ref_tokens[i - 1] == candidate[j - 1]:
                row[j] = diagonal + 1
            elif row[j - 1] > row[j]:
                row[j] = row[j - 1]
            diagonal = above
    return Int(row[candidate_len])


@export("mrs_lcs_indices")
def mrs_lcs_indices(
    ref_addr: Int,
    ref_len: Int,
    candidate_addr: Int,
    candidate_len: Int,
    table_addr: Int,
    indices_addr: Int,
) abi("C") -> Int:
    if ref_len < 0 or candidate_len < 0:
        return -1
    if ref_len == 0 or candidate_len == 0:
        return 0
    if (
        ref_addr == 0
        or candidate_addr == 0
        or table_addr == 0
        or indices_addr == 0
    ):
        return -1

    var ref_tokens = IPtr(unsafe_from_address=ref_addr)
    var candidate = IPtr(unsafe_from_address=candidate_addr)
    var table = IPtr(unsafe_from_address=table_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var cols = candidate_len + 1

    for j in range(cols):
        table[j] = 0
    for i in range(1, ref_len + 1):
        table[i * cols] = 0
        for j in range(1, candidate_len + 1):
            var pos = i * cols + j
            if ref_tokens[i - 1] == candidate[j - 1]:
                table[pos] = table[(i - 1) * cols + j - 1] + 1
            else:
                var above = table[(i - 1) * cols + j]
                var left = table[pos - 1]
                table[pos] = above if above >= left else left

    var i = ref_len
    var j = candidate_len
    var length = Int(table[ref_len * cols + candidate_len])
    var write = length
    while i > 0 and j > 0:
        if ref_tokens[i - 1] == candidate[j - 1]:
            write -= 1
            indices[write] = Int64(i - 1)
            i -= 1
            j -= 1
        elif table[i * cols + j - 1] > table[(i - 1) * cols + j]:
            j -= 1
        else:
            i -= 1
    return length
