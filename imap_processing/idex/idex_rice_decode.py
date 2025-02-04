"""
Decompress IDEX raw wavelength data.

Originally written by Corinne Wuerthner.
"""

from bitstring import BitStream

# sub_frame_size is the compression block size
SUB_FRAME_SIZE = 64


def _decode_sub_frame(
    bits: BitStream,
    psel: int,
    k: int,
    n_bits: int,
) -> tuple[list[int], int]:
    """
    Decode a subframe of compressed data.

    Parameters
    ----------
    bits : BitStream
        Raw waveform bits.
    psel : int
        Predictor select value.
    k : int
        Rice parameter used to divide each re-mapped residual value into two parts.
    n_bits : int
        Expected number of bits per sample. Either 10 or 12.

    Returns
    -------
    tuple[list, int]
        Decompressed subframe as a list of integers and the number of samples.
    """
    sample_count = 0
    sub_frame_data = []

    while (sample_count < SUB_FRAME_SIZE) and bits.pos < bits.len:
        if sample_count == 0:
            # For every subframe, the first sample is always uncompressed.
            # Read warmup sample
            d1 = bits.read(n_bits).uint
            sub_frame_data.append(d1)
            sample_count += 1

            # A 'psel' value of zero assumes that every sample in the frame is equal
            # to the first sample in the frame. In this case, the first sample is
            # stored as the original NBIT binary value
            if psel == 0:
                for _ in range(SUB_FRAME_SIZE - 1):
                    sub_frame_data.append(d1)
                sample_count = SUB_FRAME_SIZE

        # A 'psel' value of 1 assumes that the data is not well correlated, and
        # simply stores every sample in the frame using the original NBIT binary
        # representation.
        # A 'psel' value of 3 requires two uncompressed 'warm-up' samples.
        elif (psel == 1) or ((sample_count == 1) and (psel == 3)):
            d1 = bits.read(n_bits).uint
            sub_frame_data.append(d1)
            sample_count += 1

        else:
            # The rice parameter (k) is used to divide each re-mapped residual value
            # into two parts. The least significant k bits of the value are called
            # the remainder (r). The other part of the value (not included in the
            # remainder) is called the quotient (q).
            # The remapped quotient is unary encoded by including a number of 0 bits
            # equal to the value of the quotient, followed by a single '1' bit.
            q = 0
            while bits.read(1).uint == 0:
                q += 1

            if q & 0x1:
                q = int(-((q + 1) / 2))
            else:
                q = int(q / 2)

            # If the value of the quotient is equal to or larger than 47, then a
            # special symbol is used to denote that this particular residual value
            # is not rice encoded, but that this special symbol is followed by the
            # raw binary representation of the residual value using a (N_BITS+2)
            # bit binary number. This special symbol is simply 47 zeros followed
            # by a one.
            r = bits.read(k + 1).uint
            d1 = bits.read(n_bits + 2).int if q == 47 else (q << (k + 1)) + r

            if psel == 2:
                d1 = d1 + sub_frame_data[sample_count - 1]
            elif (sample_count > 1) and (psel == 3):
                d1 = (
                    d1
                    + 2 * sub_frame_data[(sample_count - 1)]
                    - sub_frame_data[(sample_count - 2)]
                )

            if (d1 > 2**n_bits) or (d1 < -(2**n_bits)):
                raise ValueError(
                    f"Overflow Error while decoding subframe "
                    f"k = {k}, q = {q}, r = {r}, d1 = {d1}\n"
                    f"DataOut = {sub_frame_data}"
                )

            sub_frame_data.append(d1)
            sample_count += 1

    return sub_frame_data, sample_count


def idex_rice_decode(
    compressed_data: str, nbit10: bool, sample_count: int
) -> list[int]:
    """
    Decode compressed IDEX wavelength data using linear prediction and Golomb-RICE.

    Parameters
    ----------
    compressed_data : str
        Binary string representation of the raw waveform.
    nbit10 : bool
        If nbit10 is true, then the samples are expected to be 10 bits each, and if
        nbit10 is false, then the samples are expected to be 12 bits each.
    sample_count : int
        The total number of samples to be decompressed.

    Returns
    -------
    list
        Decompressed data as a list of integers.
    """
    # Constants:
    k_bits = 4
    n_bits = 10 if nbit10 else 12

    # frame_size is the expected amount of data
    frame_size = sample_count
    sub_frame_per_frame = frame_size / SUB_FRAME_SIZE

    byte_data = bytearray()
    # Process 8 bits at a time
    for i in range(0, len(compressed_data), 8):
        byte_str = compressed_data[i : i + 8]
        # Convert binary string to integer
        byte_val = int(byte_str, 2)
        byte_data.append(byte_val)

    bits = BitStream(byte_data)
    out_data = []
    total_count = 0
    sub_frame_count = 0

    while bits.pos < bits.len and (sub_frame_count < sub_frame_per_frame):
        # The next two bits are the predictor select bits
        psel = bits.read(2).uint

        if psel > 1:
            k = bits.read(k_bits).uint
        else:
            k = 0

        sub_frame_data, sample_count = _decode_sub_frame(bits, psel, k, n_bits)

        out_data.extend(sub_frame_data)
        total_count += sample_count
        sub_frame_count += 1

    if bits.pos < bits.len and (len(out_data) < frame_size):
        raise ValueError("End of file reached before", frame_size, "samples decoded")

    return out_data
