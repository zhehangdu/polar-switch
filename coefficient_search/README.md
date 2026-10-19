# Coefficient search

This directory contains the CPU search and verification of circuits with two square buffers (paper Appendix A).

## Run

Install the dependencies:

```bash
python -m pip install numpy==2.5.3 scipy==1.18.1
```

Run the commands in this directory. The commands use 48 workers. Change `--workers 48` to the number of CPUs available for the run. Each run finds, scales, and verifies the circuit of one outer iteration. A `(6,4)` method has two circuits: the first circuit has 6 inner iterations, and the second circuit has 4.

First, find the circuit with inner depth 6 for singular values in $[0.001, 1]$:

```bash
python two_buffer.py --lower .001 --depth 6 --safety 1.005 \
  --seed 0 --workers 48 --output runs/stage1
```

Then find the circuit with inner depth 4. Its input interval is the output interval of the first circuit. `--center` centers its final interval at one:

```bash
python two_buffer.py --from runs/stage1.json --depth 4 --center \
  --seed 1 --workers 48 --output runs/stage2
```

`--from` verifies the first circuit again and uses the exact endpoints of its output interval. With 48 workers on an AMD EPYC 9374F node, the two searches take approximately 7 minutes and 1 minute. To verify a saved circuit again, use this command:

```bash
python two_buffer.py verify --input runs/stage1.json
```

This directory contains only the two-buffer search. It does not contain the three-buffer search or the schedule selection of the paper.

## Options

<div align="center">

| Option | Description |
| :---: | :---: |
| `--depth K` | Number of inner iterations in the circuit. Necessary for a search. |
| `--lower`, `--upper` | Endpoints of the input interval. The defaults are `.001` and `1`. |
| `--from FILE` | Use the verified output interval and the safety factor of a saved circuit as the input. Do not use it with `--lower`, `--upper`, or `--safety`. |
| `--safety` | Safety factor. The default is `1.005`. |
| `--center` | Center the final interval at one. Use it for the last outer iteration. |
| `--seed` | Nonnegative random seed. The default is `0`. |
| `--workers` | Number of CPU workers. The default is `1`. Each worker uses one thread. |
| `--output PREFIX` | Write `PREFIX.json` and `PREFIX.details.json`. Necessary for a search. The run does not overwrite files. |
| `--input FILE` | Saved circuit file (`PREFIX.json` or `PREFIX.details.json`). Necessary for `verify`. |
| `--verbose` | Show the progress of the run. |

</div>

The default search budget finds the included circuits. To see all options, including the budget, run `python two_buffer.py search --help`.

## Output

`PREFIX.json` contains the verified circuit. Each row of `iteration.steps` is one inner iteration: `write <- a * square @ square + b * addend + c * I`, where buffer `A` (`B[0]`) starts with the Gram matrix. Read all operands before you write the destination. The coefficients include all scaling. `PREFIX.details.json` records the search and the verification. The two commands above made [stage1.json](stage1.json) and [stage2.json](stage2.json).

The verification uses interval arithmetic and branch and bound over the whole input interval. It proves the lower endpoint for the exact circuit and an upper bound on the output under all rounding factors in `[1/safety, safety]`. It does not model every rounding operation on the GPU, and it does not prove that the coefficients are globally optimal. Exit code `0` means that the verification is successful; `2` means that the run did not make a verified circuit.
