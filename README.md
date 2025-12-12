# ternary_addition_reducer


## Overview

Ternary addition reducer is a high-performance tool for optimizing the number of arithmetic additions in fast matrix multiplication algorithms with ternary coefficients.
It implements multiple heuristic strategies to find near-optimal computation schemes, significantly reducing the additive cost of matrix multiplications schemes.


## Key features

* Multiple reduction strategies;
* Parallel optimization using multiple independent reducers;
* Iterative improvement with partial initialization;
* Customizable parameters for fine-tuning the optimization process;
* Comprehensive reporting with detailed statistics.

## Quick start

### Installation

```bash
git clone https://github.com/dronperminov/ternary_addition_reducer
cd ternary_addition_reducer
make -j$(nproc)
```

### Basic usage

```bash
./ternary_addition_reducer -i input_scheme.txt -o optimized_schemes
```

### Example

```bash
# Optimize a 4x7x8 matrix multiplication scheme
./ternary_addition_reducer \
    -i scheme_4x7x8.txt \
    -o optimized_schemes \
    --count 16 \
    --seed 42 \
    --max-no-improvements 5
```

## Understanding the Output

The tool provides detailed iteration reports showing progress across three independent reducing tasks (`U`, `V`, `W`):

```
+--------------------------------------------------------------------------------------------------------+
| Size: 4x7x8                  Reducers count: 32                                          Iteration: 42 |
| Rank: 164                    Naive additions: 1505                                   Elapsed: 00:02:34 |
+============================+============================+============================+=================+
|         Reducers U         |         Reducers V         |         Reducers W         |      Total      |
+----------+---------+-------+----------+---------+-------+----------+---------+-------+---------+-------+
| strategy | reduced | fresh | strategy | reduced | fresh | strategy | reduced | fresh | reduced | fresh |
+----------+---------+-------+----------+---------+-------+----------+---------+-------+---------+-------+
| gr (32)        170      85 | gr (22)        238     115 | gi (32)        309     167 |     717     367 |
| gi (45)        170      85 | ga             239     113 | wr             309     167 |     718     365 |
| gi (12)        170      85 | gi (29)        244     110 | gi (29)        310     166 |     724     361 |
| ga             171      84 | gi (43)        245      85 | mix            311     165 |     727     334 |
| gi (22)        171      84 | gi (3)         245     114 | ga             312     164 |     728     362 |
| ga             171      84 | ga             246      85 | ga             312     164 |     729     333 |
| ga             171      84 | wr             246     117 | gi (34)        312     164 |     729     365 |
| ga             172      83 | gi (12)        247      86 | gi (10)        313     163 |     732     332 |
| gi (3)         172      83 | gi (20)        247     101 | gi (3)         314     162 |     733     346 |
| gi (43)        172      83 | gi (6)         248      86 | gi (1)         316     162 |     736     331 |
+----------------------------+----------------------------+----------------------------+-----------------+
- iteration time (last / min / max / mean): 3.131 / 2.287 / 5.184 / 3.576
- best additions (U / V / W / total): 170 / 236 / 309 / 715
- best fresh vars (U / V / W / total): 85 / 117 / 167 / 369
- best strategies (U / V / W): gp (14) / gi (38) / gi (17)
```

**Columns explained**:

* `naive`: number of additions in the original scheme;
* `strategy`: optimization strategy used (see Strategies section below);
* `reduced`: number of additions after optimization;
* `fresh`: number of new variables introduced.


## Command Line Arguments

### Required arguments
* `-i PATH`: path to input scheme (plain text format)

### Optional arguments
* `-o PATH`: output directory for optimized schemes (default: `schemes`);
* `--count N`: number of parallel reducers (default: `8`);
* `--part-initialization-rate R`: probability of partial initialization from best solution (default: `0.3`);
* `--start-additions N`: upper bound for optimality check (default: `0`);
* `--max-no-improvements N`: maximum iterations without improvement (default: `3`);
* `--top-count N`: number of top reducers to display (default: `10`);
* `--seed N`: random seed for reproducibility.

#### Strategy weights
* `--ga-weight W`: greedy alternative strategy weight (default: `0.25`);
* `--gr-weight W`: greedy random strategy weight (default: `0.1`);
* `--wr-weight W`: weighted random strategy weight (default: `0.1`);
* `--gi-weight W`: greedy intersections strategy weight (default: `0.5`);
* `--gp-weight W`: greedy potential strategy weight (default: `0.0`, not used);
* `--mix-weight W`: mixed strategy weight (default: `0.05`).


## Input format
The input format is a simple plain text file with a specific structure for representing fast matrix multiplication schemes.

### Basic structure
```text
n1 n2 n3 rank
U coefficients
V coefficients
W coefficients
```

* `n1`: first matrix rows;
* `n2`: first matrix columns / second matrix rows;
* `n3`: second matrix columns;
* `rank`: number of multiplications in the scheme (tensor rank);
* `U coefficients`: space-separated ternary coefficients (`-1`, `1,` `0`) for the `U` tensor (`rank × n1 × n2` values);
* `V coefficients`: space-separated ternary coefficients (`-1`, `1,` `0`) for the `V` tensor (`rank × n2 × n3` values);
* `W coefficients`: space-separated ternary coefficients (`-1`, `1,` `0`) for the `W` tensor (`rank × n3 × n1` values).

### Example: Strassen's 2x2x2 algorithm (rank 7)
```text
2 2 2 7
0 0 0 1 0 0 1 1 0 1 0 0 1 0 0 0 1 0 1 0 1 0 1 1 1 1 1 1
1 -1 -1 1 1 -1 0 0 0 0 0 1 0 1 0 0 1 0 -1 0 1 -1 -1 0 0 0 1 0
0 0 0 1 -1 1 0 0 0 0 1 0 1 -1 1 -1 0 1 0 1 1 -1 0 -1 1 0 0 0
```


## Output format
The optimized schemes are saved in JSON format, which is fully compatible with the [FastMatrixMultiplication](https://github.com/dronperminov/FastMatrixMultiplication?tab=readme-ov-file#reduced-scheme-format) repository.
This format allows for easy integration with other tools, verification of correctness, and further processing.

## Optimization strategies
The tool employs seven different strategies:

### Greedy (`g`)
Selects the first subexpression with the highest frequency (maximum occurrences). Pure deterministic approach that always picks the most common subexpression.

### Greedy alternative (`ga`)
Randomly selects from all subexpressions with the highest frequency. If only one such subexpression exists, it behaves identically to the `Greedy` strategy.
Provides some randomness while maintaining focus on high-frequency candidates.

### Weighted random (`wr`)
Selects subexpressions randomly with weights proportional to their profit, where profit = (frequency - 1). This gives higher probability to subexpressions that
appear more frequently, but allows exploration of lower-frequency options.

### Greedy random (`gr`)
Hybrid strategy that:
* With probability `p > 50%`: uses greedy alternative;
* Otherwise: uses weighted random.

Balances exploitation (`greedy alternative`) with exploration (`weighted random`).

### Greedy potential (`gp`)
Implementation based on the article ["The Number of the Beast: Reducing Additions in Fast Matrix Multiplication Algorithms for Dimensions up to 666"](https://eprint.iacr.org/2024/2063.pdf).
For each candidate subexpression:

* Temporarily replace it with a fresh variable;
* Count profits newly created pairs (call this "potential");
* Revert the replacement to maintain correctness;
* Select subexpression with maximum: `profit + α × potential`.

Complexity: `O(|expressions| × |variables|²)` - very effective but computationally expensive.

### Greedy intersections (`gi`)

Improved version of `greedy potential` that accelerates the computation. Instead of performing actual replacements, it:

* Estimates potential by analyzing intersections between subexpressions;
* Uses weighted coefficients based on how much subexpressions intersect;
* Computes scores faster without temporary modifications.

Complexity: `O(|variables|²)` - significantly faster than `greedy potential` while maintaining good quality.

### Mix (`mix`)
Dynamically switches between all available strategies according to configured weights, allowing the algorithm to adapt its approach based on what works best for the current state.


## Algorithm Overview
The core reduction algorithm follows this iterative process:

### Step 1: Frequency Counting
For all expressions in the current scheme, count frequencies of all possible subexpressions of the form `ai + aj` or `ai - aj`.
This identifies common computational patterns that can be factored out.

### Step 2: Subexpression Selection
Using one of the strategies described above, select a candidate subexpression.

### Step 3: Replacement
Replace all occurrences of the selected subexpression with a fresh variable and update all affected expressions.

### Step 4: Iteration
Repeat steps 1-3 until no more profitable subexpressions exist (`frequency ≤ 1` for all pairs).
