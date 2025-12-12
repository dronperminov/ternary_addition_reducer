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
+----------------------------------------------------------------------------------------------------------------------------------------+
| Size: 4x7x8                          Reducers: 128                                                             Iteration: 307          |
| Rank: 164                                                                                                      Elapsed: 01:10:25       |
+====================================+====================================+====================================+=========================+
|             Reducers U             |             Reducers V             |             Reducers W             |          Total          |
+----------+-------+---------+-------+----------+-------+---------+-------+----------+-------+---------+-------+-------+---------+-------|
| strategy | naive | reduced | fresh | strategy | naive | reduced | fresh | strategy | naive | reduced | fresh | naive | reduced | fresh |
+----------+-------+---------+-------+----------+-------+---------+-------+----------+-------+---------+-------+-------+---------+-------+
| gi (26)      446       158      88 | gi (43)      500       228     115 | ga           559       307     168 |  1505       693     371 | 
| gi (5)       446       158      88 | gr (11)      500       228     117 | gi (22)      559       309     166 |  1505       695     371 | 
| gi (30)      446       158      88 | ga           500       229     113 | gr (8)       559       309     166 |  1505       696     367 | 
| gi (35)      446       158      88 | gi (48)      500       229     113 | wr           559       309     166 |  1505       696     367 | 
| gi (24)      446       158      88 | gi (13)      500       230     110 | wr           559       309     166 |  1505       697     364 | 
| ga           446       158      88 | gi (43)      500       230     112 | gi (8)       559       310     165 |  1505       698     365 | 
| gi (9)       446       158      88 | ga           500       231     108 | gr (9)       559       311     164 |  1505       700     360 | 
| gi (21)      446       158      88 | gi (38)      500       231     110 | ga           559       311     164 |  1505       700     362 | 
| gi (2)       446       158      88 | gr (48)      500       231     113 | gi (16)      559       311     164 |  1505       700     365 | 
| ga           446       158      88 | gi (42)      500       231     115 | ga           559       311     164 |  1505       700     367 | 
+------------------------------------+------------------------------------+------------------------------------+-------------------------+
- iteration time (last / min / max / mean): 10.647 / 6.262 / 00:01:30 / 13.761
- best additions (U / V / W / total): 158 / 227 / 305 / 690
- best fresh vars (U / V / W / total): 88 / 117 / 170 / 375
- best strategies (U / V / W): gi (42) / gr (13) / gp (0)
```

**Columns explained**:

* `strategy`: optimization strategy used (see Strategies section below);
* `naive`: number of additions in the original scheme;
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

### Greedy Potential (`gp`)
Implementation based on the article ["The Number of the Beast: Reducing Additions in Fast Matrix Multiplication Algorithms for Dimensions up to 666"](https://eprint.iacr.org/2024/2063.pdf).
For each candidate subexpression:

* Temporarily replace it with a fresh variable;
* Count profits newly created pairs (call this "potential");
* Revert the replacement to maintain correctness;
* Select subexpression with maximum: `profit + α × potential`.

Complexity: `O(|expressions| × |variables|²)` - very effective but computationally expensive.

### Greedy Intersections (`gi`)

Improved version of `greedy potential` that accelerates the computation. Instead of performing actual replacements, it:

* Estimates potential by analyzing intersections between subexpressions;
* Uses weighted coefficients based on how much subexpressions intersect;
* Computes scores faster without temporary modifications.

Complexity: `O(|variables|²)` - significantly faster than `greedy potential` while maintaining good quality.

### Mix (mix)
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
