#Library imports
import hashlib
import argparse
import pickle
import time
import sys
import os
import re
import itertools
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import cuda
import numpy as np

# SHA256 Constants
# First 32 bits of the fractional parts of the cube roots of the first 64 primes
K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
], dtype=np.uint32)

# Initial hash values - first 32 bits of fractional parts of square roots of first 8 primes 
H0 = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
], dtype=np.uint32)


@cuda.jit
def sha256_kernel(words_array, word_lengths, hashes_array, target, found):
    idx = cuda.grid(1)
    if idx >= words_array.shape[0]:
        return
    if found[0]:
        return

    # SHA256 constants
    K0  = np.uint32(0x428a2f98); K1  = np.uint32(0x71374491)
    K2  = np.uint32(0xb5c0fbcf); K3  = np.uint32(0xe9b5dba5)
    K4  = np.uint32(0x3956c25b); K5  = np.uint32(0x59f111f1)
    K6  = np.uint32(0x923f82a4); K7  = np.uint32(0xab1c5ed5)
    K8  = np.uint32(0xd807aa98); K9  = np.uint32(0x12835b01)
    K10 = np.uint32(0x243185be); K11 = np.uint32(0x550c7dc3)
    K12 = np.uint32(0x72be5d74); K13 = np.uint32(0x80deb1fe)
    K14 = np.uint32(0x9bdc06a7); K15 = np.uint32(0xc19bf174)
    K16 = np.uint32(0xe49b69c1); K17 = np.uint32(0xefbe4786)
    K18 = np.uint32(0x0fc19dc6); K19 = np.uint32(0x240ca1cc)
    K20 = np.uint32(0x2de92c6f); K21 = np.uint32(0x4a7484aa)
    K22 = np.uint32(0x5cb0a9dc); K23 = np.uint32(0x76f988da)
    K24 = np.uint32(0x983e5152); K25 = np.uint32(0xa831c66d)
    K26 = np.uint32(0xb00327c8); K27 = np.uint32(0xbf597fc7)
    K28 = np.uint32(0xc6e00bf3); K29 = np.uint32(0xd5a79147)
    K30 = np.uint32(0x06ca6351); K31 = np.uint32(0x14292967)
    K32 = np.uint32(0x27b70a85); K33 = np.uint32(0x2e1b2138)
    K34 = np.uint32(0x4d2c6dfc); K35 = np.uint32(0x53380d13)
    K36 = np.uint32(0x650a7354); K37 = np.uint32(0x766a0abb)
    K38 = np.uint32(0x81c2c92e); K39 = np.uint32(0x92722c85)
    K40 = np.uint32(0xa2bfe8a1); K41 = np.uint32(0xa81a664b)
    K42 = np.uint32(0xc24b8b70); K43 = np.uint32(0xc76c51a3)
    K44 = np.uint32(0xd192e819); K45 = np.uint32(0xd6990624)
    K46 = np.uint32(0xf40e3585); K47 = np.uint32(0x106aa070)
    K48 = np.uint32(0x19a4c116); K49 = np.uint32(0x1e376c08)
    K50 = np.uint32(0x2748774c); K51 = np.uint32(0x34b0bcb5)
    K52 = np.uint32(0x391c0cb3); K53 = np.uint32(0x4ed8aa4a)
    K54 = np.uint32(0x5b9cca4f); K55 = np.uint32(0x682e6ff3)
    K56 = np.uint32(0x748f82ee); K57 = np.uint32(0x78a5636f)
    K58 = np.uint32(0x84c87814); K59 = np.uint32(0x8cc70208)
    K60 = np.uint32(0x90befffa); K61 = np.uint32(0xa4506ceb)
    K62 = np.uint32(0xbef9a3f7); K63 = np.uint32(0xc67178f2)

    # Initial hash values
    h0 = np.uint32(0x6a09e667); h1 = np.uint32(0xbb67ae85)
    h2 = np.uint32(0x3c6ef372); h3 = np.uint32(0xa54ff53a)
    h4 = np.uint32(0x510e527f); h5 = np.uint32(0x9b05688c)
    h6 = np.uint32(0x1f83d9ab); h7 = np.uint32(0x5be0cd19)
    
    #Message Padding
    msg_len = word_lengths[idx]
    if msg_len > 55:
        return
    # Copy word into a 64-byte block
    block = cuda.local.array(64, dtype=np.uint8)
    for i in range(64):
        block[i] = np.uint8(0)
    for i in range(msg_len):
        block[i] = words_array[idx, i]
    
    # Append the 0x80 byte (the 1 bit)
    block[msg_len] = np.uint8(0x80)
    
    # Append message length in bits as 64-bit big-endian at the end
    bit_len = np.uint64(msg_len * 8)
    block[63] = np.uint8(bit_len & np.uint64(0xFF))
    block[62] = np.uint8((bit_len >> np.uint64(8))  & np.uint64(0xFF))
    block[61] = np.uint8((bit_len >> np.uint64(16)) & np.uint64(0xFF))
    block[60] = np.uint8((bit_len >> np.uint64(24)) & np.uint64(0xFF))
    block[59] = np.uint8((bit_len >> np.uint64(32)) & np.uint64(0xFF))
    block[58] = np.uint8((bit_len >> np.uint64(40)) & np.uint64(0xFF))
    block[57] = np.uint8((bit_len >> np.uint64(48)) & np.uint64(0xFF))
    block[56] = np.uint8((bit_len >> np.uint64(56)) & np.uint64(0xFF))
    
    # Message Scheduling
    w = cuda.local.array(64, dtype=np.uint32)
    
    # First 16 words come from the block, 4 bytes each, big-endian
    for i in range(16):
        w[i] = (np.uint32(block[i*4]) << np.uint32(24)) | \
               (np.uint32(block[i*4+1]) << np.uint32(16)) | \
               (np.uint32(block[i*4+2]) << np.uint32(8))  | \
               (np.uint32(block[i*4+3]))
    
    # Words 16-63 are computed from previous words
    for i in range(16, 64):
        s0 = ((((w[i-15] >> np.uint32(7))  | (w[i-15] << np.uint32(25))) & np.uint32(0xFFFFFFFF)) ^
              (((w[i-15] >> np.uint32(18)) | (w[i-15] << np.uint32(14))) & np.uint32(0xFFFFFFFF)) ^
              (w[i-15] >> np.uint32(3)))
        s1 = ((((w[i-2] >> np.uint32(17)) | (w[i-2] << np.uint32(15))) & np.uint32(0xFFFFFFFF)) ^
              (((w[i-2] >> np.uint32(19)) | (w[i-2] << np.uint32(13))) & np.uint32(0xFFFFFFFF)) ^
              (w[i-2] >> np.uint32(10)))
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & np.uint32(0xFFFFFFFF)
        
    K_vals = cuda.local.array(64, dtype=np.uint32)
    K_vals[0]=K0;   K_vals[1]=K1;   K_vals[2]=K2;   K_vals[3]=K3
    K_vals[4]=K4;   K_vals[5]=K5;   K_vals[6]=K6;   K_vals[7]=K7
    K_vals[8]=K8;   K_vals[9]=K9;   K_vals[10]=K10; K_vals[11]=K11
    K_vals[12]=K12; K_vals[13]=K13; K_vals[14]=K14; K_vals[15]=K15
    K_vals[16]=K16; K_vals[17]=K17; K_vals[18]=K18; K_vals[19]=K19
    K_vals[20]=K20; K_vals[21]=K21; K_vals[22]=K22; K_vals[23]=K23
    K_vals[24]=K24; K_vals[25]=K25; K_vals[26]=K26; K_vals[27]=K27
    K_vals[28]=K28; K_vals[29]=K29; K_vals[30]=K30; K_vals[31]=K31
    K_vals[32]=K32; K_vals[33]=K33; K_vals[34]=K34; K_vals[35]=K35
    K_vals[36]=K36; K_vals[37]=K37; K_vals[38]=K38; K_vals[39]=K39
    K_vals[40]=K40; K_vals[41]=K41; K_vals[42]=K42; K_vals[43]=K43
    K_vals[44]=K44; K_vals[45]=K45; K_vals[46]=K46; K_vals[47]=K47
    K_vals[48]=K48; K_vals[49]=K49; K_vals[50]=K50; K_vals[51]=K51
    K_vals[52]=K52; K_vals[53]=K53; K_vals[54]=K54; K_vals[55]=K55
    K_vals[56]=K56; K_vals[57]=K57; K_vals[58]=K58; K_vals[59]=K59
    K_vals[60]=K60; K_vals[61]=K61; K_vals[62]=K62; K_vals[63]=K63
        
    
    # Compression rounds
    a = h0; b = h1; c = h2; d = h3
    e = h4; f = h5; g = h6; h = h7

    for i in range(64):
        # Sigma functions on e
        S1 = ((((e >> np.uint32(6))  | (e << np.uint32(26))) & np.uint32(0xFFFFFFFF)) ^
      (((e >> np.uint32(11)) | (e << np.uint32(21))) & np.uint32(0xFFFFFFFF)) ^
      (((e >> np.uint32(25)) | (e << np.uint32(7)))  & np.uint32(0xFFFFFFFF)))
        
        # Choice function - if e then f else g
        ch = (e & f) ^ ((e ^ np.uint32(0xFFFFFFFF)) & g)
        
        # Sigma functions on a
        S0 = ((((a >> np.uint32(2))  | (a << np.uint32(30))) & np.uint32(0xFFFFFFFF)) ^
      (((a >> np.uint32(13)) | (a << np.uint32(19))) & np.uint32(0xFFFFFFFF)) ^
      (((a >> np.uint32(22)) | (a << np.uint32(10))) & np.uint32(0xFFFFFFFF)))
        
        # Majority function - majority vote of a, b, c
        maj = (a & b) ^ (a & c) ^ (b & c)

        # Compute temp values
        temp1 = (h + S1 + ch + K_vals[i] + w[i]) & np.uint32(0xFFFFFFFF)
        temp2 = (S0 + maj) & np.uint32(0xFFFFFFFF)

        # Rotate variables
        h = g
        g = f
        f = e
        e = (d + temp1) & np.uint32(0xFFFFFFFF)
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & np.uint32(0xFFFFFFFF)
    # Final hash assembly
    h0 = (h0 + a) & np.uint32(0xFFFFFFFF)
    h1 = (h1 + b) & np.uint32(0xFFFFFFFF)
    h2 = (h2 + c) & np.uint32(0xFFFFFFFF)
    h3 = (h3 + d) & np.uint32(0xFFFFFFFF)
    h4 = (h4 + e) & np.uint32(0xFFFFFFFF)
    h5 = (h5 + f) & np.uint32(0xFFFFFFFF)
    h6 = (h6 + g) & np.uint32(0xFFFFFFFF)
    h7 = (h7 + h) & np.uint32(0xFFFFFFFF)

    # Convert to bytes and store in hashes_array
    # Each h value is 4 bytes, big-endian
    for i in range(4):
        hashes_array[idx, 0*4+i]  = np.uint8((h0 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 1*4+i]  = np.uint8((h1 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 2*4+i]  = np.uint8((h2 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 3*4+i]  = np.uint8((h3 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 4*4+i]  = np.uint8((h4 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 5*4+i]  = np.uint8((h5 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 6*4+i]  = np.uint8((h6 >> np.uint32(24 - i*8)) & np.uint32(0xFF))
        hashes_array[idx, 7*4+i]  = np.uint8((h7 >> np.uint32(24 - i*8)) & np.uint32(0xFF))

    # Compare to target
    match = True
    for i in range(32):
        if hashes_array[idx, i] != target[i]:
            match = False
            break
    
    if match:
        found[0] = True

'''def debug_kernel():
    test_word = "admin123"
    expected = hashlib.sha256(test_word.encode()).hexdigest()
    print(f"Expected: {expected}")
    
    # Build single-word input
    encoded = test_word.encode()
    n = 1
    max_len = len(encoded)
    
    words_array = np.zeros((n, max_len), dtype=np.uint8)
    word_lengths = np.zeros(n, dtype=np.int32)
    
    for j, b in enumerate(encoded):
        words_array[0, j] = b
    word_lengths[0] = len(encoded)
    
    hashes_array = np.zeros((n, 32), dtype=np.uint8)
    target_bytes = np.array([int(expected[i:i+2], 16) for i in range(0, 64, 2)], dtype=np.uint8)
    found = np.zeros(1, dtype=np.bool_)
    
    d_words = cuda.to_device(words_array)
    d_lengths = cuda.to_device(word_lengths)
    d_hashes = cuda.to_device(hashes_array)
    d_target = cuda.to_device(target_bytes)
    d_found = cuda.to_device(found)
    
    sha256_kernel[1, 1](d_words, d_lengths, d_hashes, d_target, d_found)
    
    d_hashes.copy_to_host(hashes_array)
    d_found.copy_to_host(found)
    
    gpu_result = ''.join(f'{b:02x}' for b in hashes_array[0])
    print(f"GPU got:  {gpu_result}")
    print(f"Match: {found[0]}")
    print(f"Kernel correct: {gpu_result == expected}")

debug_kernel()'''

def gpu_sha256_crack(target_hash, wordlist_path):
    try:
        cuda.detect()
    except Exception:
        print("No GPU detected, falling back to CPU dictionary attack")
        return dictionary_attack(target_hash, wordlist_path, "sha256")

    # Convert target hash from hex string to bytes array
    target_bytes = np.array([int(target_hash[i:i+2], 16) 
                             for i in range(0, 64, 2)], dtype=np.uint8)

    # Load wordlist
    with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
        words = [line.strip() for line in f if line.strip()]

    # Process in batches to avoid running out of GPU memory
    batch_size = 100000
    
    for batch_start in range(0, len(words), batch_size):
        batch = words[batch_start:batch_start + batch_size]
        n = len(batch)
        max_len = max(len(w.encode('utf-8', errors='ignore')) for w in batch)

        # Build 2D array of words as bytes, padded to max_len
        words_array = np.zeros((n, max_len), dtype=np.uint8)
        word_lengths = np.zeros(n, dtype=np.int32)
        
        for i, word in enumerate(batch):
            encoded = word.encode('utf-8', errors='ignore')
            word_lengths[i] = len(encoded)
            for j, b in enumerate(encoded):
                words_array[i, j] = b

        # Output array for hashes
        hashes_array = np.zeros((n, 32), dtype=np.uint8)
        
        # Found flag
        found = np.zeros(1, dtype=np.bool_)

        # Send to GPU
        d_words = cuda.to_device(words_array)
        d_lengths = cuda.to_device(word_lengths)
        d_hashes = cuda.to_device(hashes_array)
        d_target = cuda.to_device(target_bytes)
        d_found = cuda.to_device(found)

        # Launch kernel
        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block
        sha256_kernel[blocks, threads_per_block](
            d_words, d_lengths, d_hashes, d_target, d_found
        )
        cuda.synchronize()
        
        # Get results back
        d_found.copy_to_host(found)
        
        if found[0]:
            d_hashes.copy_to_host(hashes_array)
            # Find which word matched
            for i in range(n):
                h = hashlib.sha256(batch[i].encode()).hexdigest()
                if h == target_hash:
                    return batch[i]

    return None

# Printable ASCII charset: chars 32-126 = 95 characters
CHARSET_SIZE = 95
CHARSET_OFFSET = 32  # ord(' ') = 32, printable ASCII starts here

@cuda.jit(device=True)
def sha256_device(block, digest):
    K0  = np.uint32(0x428a2f98); K1  = np.uint32(0x71374491)
    K2  = np.uint32(0xb5c0fbcf); K3  = np.uint32(0xe9b5dba5)
    K4  = np.uint32(0x3956c25b); K5  = np.uint32(0x59f111f1)
    K6  = np.uint32(0x923f82a4); K7  = np.uint32(0xab1c5ed5)
    K8  = np.uint32(0xd807aa98); K9  = np.uint32(0x12835b01)
    K10 = np.uint32(0x243185be); K11 = np.uint32(0x550c7dc3)
    K12 = np.uint32(0x72be5d74); K13 = np.uint32(0x80deb1fe)
    K14 = np.uint32(0x9bdc06a7); K15 = np.uint32(0xc19bf174)
    K16 = np.uint32(0xe49b69c1); K17 = np.uint32(0xefbe4786)
    K18 = np.uint32(0x0fc19dc6); K19 = np.uint32(0x240ca1cc)
    K20 = np.uint32(0x2de92c6f); K21 = np.uint32(0x4a7484aa)
    K22 = np.uint32(0x5cb0a9dc); K23 = np.uint32(0x76f988da)
    K24 = np.uint32(0x983e5152); K25 = np.uint32(0xa831c66d)
    K26 = np.uint32(0xb00327c8); K27 = np.uint32(0xbf597fc7)
    K28 = np.uint32(0xc6e00bf3); K29 = np.uint32(0xd5a79147)
    K30 = np.uint32(0x06ca6351); K31 = np.uint32(0x14292967)
    K32 = np.uint32(0x27b70a85); K33 = np.uint32(0x2e1b2138)
    K34 = np.uint32(0x4d2c6dfc); K35 = np.uint32(0x53380d13)
    K36 = np.uint32(0x650a7354); K37 = np.uint32(0x766a0abb)
    K38 = np.uint32(0x81c2c92e); K39 = np.uint32(0x92722c85)
    K40 = np.uint32(0xa2bfe8a1); K41 = np.uint32(0xa81a664b)
    K42 = np.uint32(0xc24b8b70); K43 = np.uint32(0xc76c51a3)
    K44 = np.uint32(0xd192e819); K45 = np.uint32(0xd6990624)
    K46 = np.uint32(0xf40e3585); K47 = np.uint32(0x106aa070)
    K48 = np.uint32(0x19a4c116); K49 = np.uint32(0x1e376c08)
    K50 = np.uint32(0x2748774c); K51 = np.uint32(0x34b0bcb5)
    K52 = np.uint32(0x391c0cb3); K53 = np.uint32(0x4ed8aa4a)
    K54 = np.uint32(0x5b9cca4f); K55 = np.uint32(0x682e6ff3)
    K56 = np.uint32(0x748f82ee); K57 = np.uint32(0x78a5636f)
    K58 = np.uint32(0x84c87814); K59 = np.uint32(0x8cc70208)
    K60 = np.uint32(0x90befffa); K61 = np.uint32(0xa4506ceb)
    K62 = np.uint32(0xbef9a3f7); K63 = np.uint32(0xc67178f2)

    K_vals = cuda.local.array(64, dtype=np.uint32)
    K_vals[0]=K0;   K_vals[1]=K1;   K_vals[2]=K2;   K_vals[3]=K3
    K_vals[4]=K4;   K_vals[5]=K5;   K_vals[6]=K6;   K_vals[7]=K7
    K_vals[8]=K8;   K_vals[9]=K9;   K_vals[10]=K10; K_vals[11]=K11
    K_vals[12]=K12; K_vals[13]=K13; K_vals[14]=K14; K_vals[15]=K15
    K_vals[16]=K16; K_vals[17]=K17; K_vals[18]=K18; K_vals[19]=K19
    K_vals[20]=K20; K_vals[21]=K21; K_vals[22]=K22; K_vals[23]=K23
    K_vals[24]=K24; K_vals[25]=K25; K_vals[26]=K26; K_vals[27]=K27
    K_vals[28]=K28; K_vals[29]=K29; K_vals[30]=K30; K_vals[31]=K31
    K_vals[32]=K32; K_vals[33]=K33; K_vals[34]=K34; K_vals[35]=K35
    K_vals[36]=K36; K_vals[37]=K37; K_vals[38]=K38; K_vals[39]=K39
    K_vals[40]=K40; K_vals[41]=K41; K_vals[42]=K42; K_vals[43]=K43
    K_vals[44]=K44; K_vals[45]=K45; K_vals[46]=K46; K_vals[47]=K47
    K_vals[48]=K48; K_vals[49]=K49; K_vals[50]=K50; K_vals[51]=K51
    K_vals[52]=K52; K_vals[53]=K53; K_vals[54]=K54; K_vals[55]=K55
    K_vals[56]=K56; K_vals[57]=K57; K_vals[58]=K58; K_vals[59]=K59
    K_vals[60]=K60; K_vals[61]=K61; K_vals[62]=K62; K_vals[63]=K63

    w = cuda.local.array(64, dtype=np.uint32)
    for i in range(16):
        w[i] = (np.uint32(block[i*4])   << np.uint32(24)) | \
               (np.uint32(block[i*4+1]) << np.uint32(16)) | \
               (np.uint32(block[i*4+2]) << np.uint32(8))  | \
               (np.uint32(block[i*4+3]))

    for i in range(16, 64):
        s0 = ((((w[i-15] >> np.uint32(7))  | (w[i-15] << np.uint32(25))) & np.uint32(0xFFFFFFFF)) ^
              (((w[i-15] >> np.uint32(18)) | (w[i-15] << np.uint32(14))) & np.uint32(0xFFFFFFFF)) ^
              (w[i-15] >> np.uint32(3)))
        s1 = ((((w[i-2] >> np.uint32(17)) | (w[i-2] << np.uint32(15))) & np.uint32(0xFFFFFFFF)) ^
              (((w[i-2] >> np.uint32(19)) | (w[i-2] << np.uint32(13))) & np.uint32(0xFFFFFFFF)) ^
              (w[i-2] >> np.uint32(10)))
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & np.uint32(0xFFFFFFFF)

    h0 = np.uint32(0x6a09e667); h1 = np.uint32(0xbb67ae85)
    h2 = np.uint32(0x3c6ef372); h3 = np.uint32(0xa54ff53a)
    h4 = np.uint32(0x510e527f); h5 = np.uint32(0x9b05688c)
    h6 = np.uint32(0x1f83d9ab); h7 = np.uint32(0x5be0cd19)

    a = h0; b = h1; c = h2; d = h3
    e = h4; f = h5; g = h6; hh = h7

    for i in range(64):
        S1 = ((((e >> np.uint32(6))  | (e << np.uint32(26))) & np.uint32(0xFFFFFFFF)) ^
              (((e >> np.uint32(11)) | (e << np.uint32(21))) & np.uint32(0xFFFFFFFF)) ^
              (((e >> np.uint32(25)) | (e << np.uint32(7)))  & np.uint32(0xFFFFFFFF)))
        ch = (e & f) ^ ((e ^ np.uint32(0xFFFFFFFF)) & g)
        S0 = ((((a >> np.uint32(2))  | (a << np.uint32(30))) & np.uint32(0xFFFFFFFF)) ^
              (((a >> np.uint32(13)) | (a << np.uint32(19))) & np.uint32(0xFFFFFFFF)) ^
              (((a >> np.uint32(22)) | (a << np.uint32(10))) & np.uint32(0xFFFFFFFF)))
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp1 = (hh + S1 + ch + K_vals[i] + w[i]) & np.uint32(0xFFFFFFFF)
        temp2 = (S0 + maj) & np.uint32(0xFFFFFFFF)
        hh = g; g = f; f = e
        e = (d + temp1) & np.uint32(0xFFFFFFFF)
        d = c; c = b; b = a
        a = (temp1 + temp2) & np.uint32(0xFFFFFFFF)

    h0 = (h0 + a) & np.uint32(0xFFFFFFFF)
    h1 = (h1 + b) & np.uint32(0xFFFFFFFF)
    h2 = (h2 + c) & np.uint32(0xFFFFFFFF)
    h3 = (h3 + d) & np.uint32(0xFFFFFFFF)
    h4 = (h4 + e) & np.uint32(0xFFFFFFFF)
    h5 = (h5 + f) & np.uint32(0xFFFFFFFF)
    h6 = (h6 + g) & np.uint32(0xFFFFFFFF)
    h7 = (h7 + hh) & np.uint32(0xFFFFFFFF)

    for i in range(4):
        shift = np.uint32(24 - i * 8)
        digest[0*4+i] = np.uint8((h0 >> shift) & np.uint32(0xFF))
        digest[1*4+i] = np.uint8((h1 >> shift) & np.uint32(0xFF))
        digest[2*4+i] = np.uint8((h2 >> shift) & np.uint32(0xFF))
        digest[3*4+i] = np.uint8((h3 >> shift) & np.uint32(0xFF))
        digest[4*4+i] = np.uint8((h4 >> shift) & np.uint32(0xFF))
        digest[5*4+i] = np.uint8((h5 >> shift) & np.uint32(0xFF))
        digest[6*4+i] = np.uint8((h6 >> shift) & np.uint32(0xFF))
        digest[7*4+i] = np.uint8((h7 >> shift) & np.uint32(0xFF))


@cuda.jit
def sha256_brute_kernel(start_idx, length, target, found, found_idx):
    idx = cuda.grid(1)
    global_idx = start_idx + idx

    if found[0]:
        return

    block = cuda.local.array(64, dtype=np.uint8)
    for i in range(64):
        block[i] = np.uint8(0)

    remaining = global_idx
    for i in range(length):
        block[i] = np.uint8((remaining % CHARSET_SIZE) + CHARSET_OFFSET)
        remaining = remaining // CHARSET_SIZE

    block[length] = np.uint8(0x80)

    bit_len = np.uint64(length * 8)
    block[56] = np.uint8((bit_len >> np.uint64(56)) & np.uint64(0xFF))
    block[57] = np.uint8((bit_len >> np.uint64(48)) & np.uint64(0xFF))
    block[58] = np.uint8((bit_len >> np.uint64(40)) & np.uint64(0xFF))
    block[59] = np.uint8((bit_len >> np.uint64(32)) & np.uint64(0xFF))
    block[60] = np.uint8((bit_len >> np.uint64(24)) & np.uint64(0xFF))
    block[61] = np.uint8((bit_len >> np.uint64(16)) & np.uint64(0xFF))
    block[62] = np.uint8((bit_len >> np.uint64(8))  & np.uint64(0xFF))
    block[63] = np.uint8(bit_len & np.uint64(0xFF))

    digest = cuda.local.array(32, dtype=np.uint8)
    sha256_device(block, digest)

    match = True
    for i in range(32):
        if digest[i] != target[i]:
            match = False
            break

    if match:
        found[0] = True
        found_idx[0] = global_idx


def gpu_brute_force(target_hash, max_length=8):
    try:
        cuda.detect()
    except Exception:
        print("No GPU detected")
        return None

    target_bytes = np.array([int(target_hash[i:i+2], 16)
                             for i in range(0, 64, 2)], dtype=np.uint8)

    d_target = cuda.to_device(target_bytes)
    found = np.zeros(1, dtype=np.bool_)
    found_idx = np.zeros(1, dtype=np.int64)
    d_found = cuda.to_device(found)
    d_found_idx = cuda.to_device(found_idx)

    batch_size = 10_000_000
    threads_per_block = 256

    for length in range(1, max_length + 1):
        total = CHARSET_SIZE ** length
        print(f"Trying length {length}: {total:,} combinations")

        for batch_start in range(0, total, batch_size):
            if found[0]:
                break

            current_batch = min(batch_size, total - batch_start)
            blocks = (current_batch + threads_per_block - 1) // threads_per_block

            sha256_brute_kernel[blocks, threads_per_block](
                np.int64(batch_start), np.int32(length),
                d_target, d_found, d_found_idx
            )
            cuda.synchronize()
            d_found.copy_to_host(found)

        if found[0]:
            d_found_idx.copy_to_host(found_idx)
            idx = int(found_idx[0])
            result = []
            for _ in range(length):
                result.append(chr((idx % CHARSET_SIZE) + CHARSET_OFFSET))
                idx //= CHARSET_SIZE
            return ''.join(result)

    return None

#argparse setup for CLI use
parser = argparse.ArgumentParser()
parser.add_argument("--hash", required=True, help="The hash to crack")
parser.add_argument("--wordlist", help="Path to wordlist file")
parser.add_argument("--algorithm", default="md5", help="Hash algorithm (default: md5)")
parser.add_argument("--build-lookup", action="store_true", help="Build a lookup table")
parser.add_argument("--lookup-file", default="lookup.pkl", help="Look table filename")
parser.add_argument("--mode", choices=["dictionary", "lookup", "build", "brute", "gpu", "gpu-brute"], default="dictionary")
parser.add_argument("--threads", type=int, default=8, help="Number of threads to use (default: 8, use 1 to disable threading)")
parser.add_argument("--max-length", type=int, default=8, help="Max password length for brute force (default: 8)")
args = parser.parse_args()

target_hash = args.hash

#Functions
 
def identify_hash(hash_str):
    hash_str = hash_str.strip().lower()
    candidates = []

    if hash_str.startswith("$2a$") or hash_str.startswith("$2b$"):
        candidates.append("bcrypt")
        return candidates

    if hash_str.startswith("$6$"):
        candidates.append("sha512crypt")
        return candidates

    if hash_str.startswith("$1$"):
        candidates.append("md5crypt")
        return candidates

    if re.match(r"^[a-f0-9]+$", hash_str):
        length = len(hash_str)
        if length == 32:
            candidates.append("md5")
        elif length == 40:
            candidates.append("sha1")
            candidates.append("ripemd160")
        elif length == 56:
            candidates.append("sha224")
        elif length == 64:
            candidates.append("sha256")
            candidates.append("sha3_256")
        elif length == 96:
            candidates.append("sha384")
            candidates.append("sha3_384")
        elif length == 128:
            candidates.append("sha512")
            candidates.append("sha3_512")
        else:
            candidates.append("unknown")

    return candidates


def hash_worker(word, algorithm):
    """Helper function to run in a separate process"""
    return hashlib.new(algorithm, word.encode()).hexdigest(), word

def dictionary_attack(hash_str, wordlist_path, algorithm, num_threads=8):
    if not os.path.isfile(wordlist_path):
        return None

    
    with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
        
        batch_size = 10000 
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            while True:
                
                words = [f.readline().strip() for _ in range(batch_size)]
                words = [w for w in words if w] 
                if not words: break

                
                futures = {executor.submit(hash_worker, w, algorithm): w for w in words}
                
                for future in as_completed(futures):
                    hashed_res, original_word = future.result()
                    if hashed_res == hash_str:
                        return original_word
    return None

def build_lookup_table(wordlist_path, algorithm):
    if not os.path.isfile(wordlist_path):
        print(f"Wordlist file '{wordlist_path}' not found.")
        return None

    unsupported = ["bcrypt", "sha512crypt", "md5crypt"]
    if algorithm in unsupported:
        print(f"{algorithm} is not supported for lookup table generation.")
        return None

    lookup_table = {}
    with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            try:
                hashed_word = hashlib.new(algorithm, word.encode()).hexdigest()
                lookup_table[hashed_word] = word
            except ValueError:
                print(f"Unsupported algorithm: {algorithm}")
                return None

    return lookup_table

def save_lookup_table(lookup_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lookup_table, f)

def load_lookup_table(filename):
    if not os.path.isfile(filename):
        print(f"Lookup table file '{filename}' not found.")
        return None
    with open(filename, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict      


def lookup_table_attack(hash_str, lookup_table):
    return lookup_table.get(hash_str, None)


def brute_force_attack(hash_str, algorithm, max_length=10, num_threads=8):
    chars = string.ascii_letters + string.digits + string.punctuation
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for length in range(1, max_length + 1):
            print(f"Checking length: {length}")
            
            combinations = itertools.product(chars, repeat=length)
            
            
            batch_size = 50000
            while True:
                
                batch_gen = itertools.islice(combinations, batch_size)
                batch = [''.join(c) for c in batch_gen]
                if not batch: 
                    break
                
                
                futures = {executor.submit(hash_worker, w, algorithm): w for w in batch}
                for future in as_completed(futures):
                    hashed_res, original_word = future.result()
                    if hashed_res == hash_str:
                        return original_word
                
                
    return None



#Main logic
if "__main__" == __name__:
    results = identify_hash(target_hash)

    if not results:
     print("Unknown hash format")
    else:
     print(f"Possible hash algorithms: {results}")

    if args.mode == "dictionary":
        if not args.wordlist:
            print("Please provide a wordlist with --wordlist")
        else:
            result = dictionary_attack(target_hash, args.wordlist, results[0], num_threads=args.threads)
            if result:
                print(f"Password found: {result}")
            else:
                print("Password not found in wordlist")

    elif args.mode == "build":
         if not args.wordlist:
            print("Please provide a wordlist with --wordlist")
         else:
            for algo in ["md5", "sha256"]:
                print(f"Building {algo} lookup table...")
                table = build_lookup_table(args.wordlist, algo)
                save_lookup_table(table, f"lookup_{algo}.pkl")
                print(f"Saved lookup_{algo}.pkl")

    elif args.mode == "lookup":
        found = False
        for algo in ["md5", "sha256"]:
            table = load_lookup_table(f"lookup_{algo}.pkl")
            if table:
                result = lookup_table_attack(target_hash, table)
                if result:
                    print(f"Password found ({algo}): {result}")
                    found = True
                    break
        if not found:
            print("Password not found in lookup table")

    elif args.mode == "brute":
        result = brute_force_attack(target_hash, results[0], num_threads=args.threads)
        if result:
            print(f"Password found: {result}")
        else:
            print("Password not found")
    elif args.mode == "gpu":
        if not args.wordlist:
            print("Please provide a wordlist with --wordlist")
        else:
            result = gpu_sha256_crack(target_hash, args.wordlist)
            if result:
                print(f"Password found: {result}")
            else:
                print("Password not found")        
    elif args.mode == "gpu-brute":
     result = gpu_brute_force(target_hash, max_length=args.max_length)
    if result:
        print(f"Password found: {result}")
    else:
        print("Password not found")            