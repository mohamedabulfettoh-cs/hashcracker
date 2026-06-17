# GPU Hash Cracker

Hashcracker is a simple hash-cracking tool written in Python. Is it a replacement to Hashcat or John the Ripper? Definitely not — but I built it for experience, to explore how hash-cracking can be implemented from scratch in Python, and to dig into GPU-accelerated computation using CUDA via Numba.

> Built with an RTX 4060 Laptop GPU and a lot of debugging sessions.

---

## Features

- **Hash identification** — Automatically detects the likely algorithm from the hash format (MD5, SHA-1, SHA-224, SHA-256, SHA-384, SHA-512, bcrypt, sha512crypt, etc.)
- **Dictionary attack** — Wordlist-based cracking using Python's `hashlib`
- **Multi-threaded CPU cracking** — Uses `ProcessPoolExecutor` for parallel dictionary and brute-force attacks
- **Lookup table mode** — Pre-build hash→plaintext tables (pickled) and query them instantly
- **CPU brute-force** — Exhaustive character-space search up to a configurable max length
- **GPU dictionary attack** — Offloads SHA-256 hash computation to the GPU via a custom CUDA kernel (`gpu` mode)
- **GPU brute-force** — Full brute-force over printable ASCII using a CUDA kernel (`gpu-brute` mode), no wordlist needed

---

## Modes

| Mode | Flag | Description |
|---|---|---|
| Dictionary (CPU) | `--mode dictionary` | Wordlist attack using `hashlib` + multiprocessing |
| Brute-force (CPU) | `--mode brute` | Exhaustive search over printable ASCII |
| Build lookup table | `--mode build` | Pre-computes MD5 + SHA-256 tables from a wordlist |
| Lookup table attack | `--mode lookup` | Queries pre-built `.pkl` lookup tables |
| GPU dictionary | `--mode gpu` | SHA-256 dictionary attack accelerated on GPU |
| GPU brute-force | `--mode gpu-brute` | SHA-256 brute-force entirely on GPU |

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (for `gpu` and `gpu-brute` modes)
- [Numba](https://numba.readthedocs.io/) (`pip install numba`)
- NumPy (`pip install numpy`)
- CUDA Toolkit installed and on PATH

---

## Installation

```bash
git clone https://github.com/mohamedabulfettoh-cs/HashCracker
cd HashCracker
pip install numba numpy
```

Make sure your CUDA toolkit is installed and `nvcc` is accessible. On Windows, you may need to manually add the CUDA bin directory to your `PATH`.

---

## Usage

```bash
python hashcracker.py --hash <HASH> --mode <MODE> [options]
```

### Options

| Flag | Description | Default |
|---|---|---|
| `--hash` | The hash to crack *(required)* | — |
| `--wordlist` | Path to wordlist file | — |
| `--algorithm` | Hash algorithm hint | `md5` |
| `--mode` | Attack mode (see table above) | `dictionary` |
| `--threads` | Number of CPU threads | `8` |
| `--max-length` | Max password length for brute-force | `8` |
| `--build-lookup` | Build a lookup table | — |
| `--lookup-file` | Filename for lookup table | `lookup.pkl` |

### Examples

**Dictionary attack (CPU):**
```bash
python hashcracker.py --hash 5f4dcc3b5aa765d61d8327deb882cf99 --wordlist rockyou.txt --mode dictionary
```

**GPU dictionary attack (SHA-256):**
```bash
python hashcracker.py --hash <sha256_hash> --wordlist rockyou.txt --mode gpu
```

**GPU brute-force (up to 5 characters):**
```bash
python hashcracker.py --hash <sha256_hash> --mode gpu-brute --max-length 5
```

**Build lookup tables:**
```bash
python hashcracker.py --hash dummy --wordlist rockyou.txt --mode build
```

**Lookup table attack:**
```bash
python hashcracker.py --hash 5f4dcc3b5aa765d61d8327deb882cf99 --mode lookup
```

---

## How the GPU Cracking Works

The GPU modes implement SHA-256 from scratch directly in CUDA using Numba's `@cuda.jit` decorator — no libraries, just the raw algorithm running on thousands of threads in parallel.

Each GPU thread handles one candidate password:
1. The password is padded into a 64-byte block following the SHA-256 spec (append `0x80`, then the message bit-length at bytes 56–63)
2. The 64 SHA-256 round constants and initial hash values are hardcoded directly in the kernel to avoid global memory access
3. Message scheduling and 64 compression rounds execute per-thread
4. The resulting digest is compared byte-by-byte against the target hash
5. A shared `found` flag short-circuits remaining threads once a match is detected

The GPU brute-force kernel additionally handles candidate generation on-device — each thread decodes its index into a character combination over printable ASCII (32–126), so no wordlist is needed at all.

---

## Limitations

- GPU modes currently support **SHA-256 only**
- Passwords longer than 55 bytes are skipped (single-block SHA-256 constraint)
- The brute-force space grows exponentially — 8-character printable ASCII is ~6.6 trillion combinations, which takes time even on a GPU
- Not a replacement for Hashcat. The goal here was learning, not performance

---

## Notes

This project was built as a learning exercise to understand:
- How SHA-256 is implemented at the byte level
- How CUDA kernels work and how to write GPU-parallel code in Python with Numba
- How hash-cracking tools are structured under the hood

---

## License

MIT

## Disclaimer

This tool was built strictly for educational purposes — to understand how hash-cracking works at an implementation level, and to explore GPU-accelerated computation using CUDA.

**Only use this tool on systems, accounts, and data that you own or have explicit written permission to test.** Unauthorized use of hash-cracking tools against systems or credentials you do not own is illegal in most jurisdictions and unethical regardless.

The author takes no responsibility for any misuse of this software.

