# Experiment summary

The table gives the final validation loss and training time of each method on one NVIDIA B200 GPU. Approximate model FLOPs utilization (MFU) is reported only for Qwen3-0.6B and Gemma 3-1B. Each loss links to its training record.

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col" align="center">Method</th>
      <th colspan="2" scope="colgroup" align="center">Modded-NanoGPT #91</th>
      <th colspan="3" scope="colgroup" align="center">Qwen3-0.6B</th>
      <th colspan="3" scope="colgroup" align="center">Gemma 3-1B</th>
    </tr>
    <tr>
      <th scope="col" align="center">Mean final loss</th>
      <th scope="col" align="center">Training time (h)</th>
      <th scope="col" align="center">Final loss</th>
      <th scope="col" align="center">Training time (h)</th>
      <th scope="col" align="center">Approx. MFU (%)</th>
      <th scope="col" align="center">Final loss</th>
      <th scope="col" align="center">Training time (h)</th>
      <th scope="col" align="center">Approx. MFU (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="center">Jordan</th>
      <td align="center"><a href="nanogpt/Jordan.csv">3.2778</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/Jordan.csv">2.6252</a></td>
      <td align="center">18.185</td>
      <td align="center">35.2</td>
      <td align="center"><a href="gemma/Jordan.csv">2.5534</a></td>
      <td align="center">43.413</td>
      <td align="center">39.7</td>
    </tr>
    <tr>
      <th scope="row" align="center">PE</th>
      <td align="center"><a href="nanogpt/PE.csv">3.2773</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PE.csv">2.6255</a></td>
      <td align="center">18.519</td>
      <td align="center">34.6</td>
      <td align="center"><a href="gemma/PE.csv">2.5523</a></td>
      <td align="center">43.444</td>
      <td align="center">39.5</td>
    </tr>
    <tr>
      <th scope="row" align="center">PE (BF16)<sup>*</sup></th>
      <td align="center"><a href="nanogpt/PE-BF16.csv">3.2772</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PE-BF16.csv">2.6260</a></td>
      <td align="center">18.555</td>
      <td align="center">35.2</td>
      <td align="center"><a href="gemma/PE-BF16.csv">2.5523</a></td>
      <td align="center">42.522</td>
      <td align="center">39.9</td>
    </tr>
    <tr>
      <th scope="row" align="center">GNS</th>
      <td align="center"><a href="nanogpt/GNS.csv">3.2774</a></td>
      <td align="center">0.169</td>
      <td align="center"><a href="qwen/GNS.csv">2.6250</a></td>
      <td align="center">18.032</td>
      <td align="center">35.4</td>
      <td align="center"><a href="gemma/GNS.csv">2.5515</a></td>
      <td align="center">43.458</td>
      <td align="center">39.4</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-55</th>
      <td align="center"><a href="nanogpt/PS2-55.csv">3.2770</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PS2-55.csv">2.6257</a></td>
      <td align="center">18.393</td>
      <td align="center">34.9</td>
      <td align="center"><a href="gemma/PS2-55.csv">2.5520</a></td>
      <td align="center">43.011</td>
      <td align="center">39.8</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-64</th>
      <td align="center"><a href="nanogpt/PS2-64.csv">3.2774</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PS2-64.csv">2.6258</a></td>
      <td align="center">18.358</td>
      <td align="center">34.8</td>
      <td align="center"><a href="gemma/PS2-64.csv">2.5524</a></td>
      <td align="center">42.922</td>
      <td align="center">39.4</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-54</th>
      <td align="center"><a href="nanogpt/PS2-54.csv">3.2772</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PS2-54.csv">2.6257</a></td>
      <td align="center">18.128</td>
      <td align="center">35.3</td>
      <td align="center"><a href="gemma/PS2-54.csv">2.5525</a></td>
      <td align="center">43.427</td>
      <td align="center">39.5</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-512</th>
      <td align="center"><a href="nanogpt/PS2-512.csv">3.2777</a></td>
      <td align="center">0.169</td>
      <td align="center"><a href="qwen/PS2-512.csv">2.6258</a></td>
      <td align="center">18.516</td>
      <td align="center">34.7</td>
      <td align="center">–</td>
      <td align="center">–</td>
      <td align="center">–</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-4111</th>
      <td align="center"><a href="nanogpt/PS2-4111.csv">3.2776</a></td>
      <td align="center">0.169</td>
      <td align="center"><a href="qwen/PS2-4111.csv">2.6257</a></td>
      <td align="center">18.446</td>
      <td align="center">34.7</td>
      <td align="center">–</td>
      <td align="center">–</td>
      <td align="center">–</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS2-11111</th>
      <td align="center"><a href="nanogpt/PS2-11111.csv">3.2771</a></td>
      <td align="center">0.169</td>
      <td align="center"><a href="qwen/PS2-11111.csv">2.6260</a></td>
      <td align="center">18.137</td>
      <td align="center">35.3</td>
      <td align="center">–</td>
      <td align="center">–</td>
      <td align="center">–</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS3-55</th>
      <td align="center"><a href="nanogpt/PS3-55.csv">3.2771</a></td>
      <td align="center">0.169</td>
      <td align="center"><a href="qwen/PS3-55.csv">2.6255</a></td>
      <td align="center">18.148</td>
      <td align="center">35.2</td>
      <td align="center"><a href="gemma/PS3-55.csv">2.5524</a></td>
      <td align="center">42.570</td>
      <td align="center">39.9</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS3-64</th>
      <td align="center"><a href="nanogpt/PS3-64.csv">3.2770</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PS3-64.csv">2.6257</a></td>
      <td align="center">18.460</td>
      <td align="center">34.6</td>
      <td align="center"><a href="gemma/PS3-64.csv">2.5522</a></td>
      <td align="center">43.525</td>
      <td align="center">39.7</td>
    </tr>
    <tr>
      <th scope="row" align="center">PS3-54</th>
      <td align="center"><a href="nanogpt/PS3-54.csv">3.2784</a></td>
      <td align="center">0.170</td>
      <td align="center"><a href="qwen/PS3-54.csv">2.6266</a></td>
      <td align="center">18.196</td>
      <td align="center">35.1</td>
      <td align="center"><a href="gemma/PS3-54.csv">2.5533</a></td>
      <td align="center">43.252</td>
      <td align="center">39.5</td>
    </tr>
  </tbody>
</table>

</div>

Modded-NanoGPT uses seeds 0–19. Its loss is the mean over the seeds, and its training time is the median. Qwen3-0.6B and Gemma 3-1B use seed 42. A dash shows a method that we did not train on Gemma 3-1B.

<sup>*</sup> PE (BF16) normalizes the input as $X/(\lVert X\rVert_F+0.01)/1.02$ and then runs the Polar Express iterations in `bfloat16`. It is a training baseline. The `polar_switch` package does not include it.

## Records

[nanogpt/](nanogpt/), [qwen/](qwen/), and [gemma/](gemma/) contain one CSV file for each method:

- **`nanogpt/`:** for each seed, the validation loss and the training time `train_time_ms` after 0, 250, 500, 750, 1,000, 1,250, and 1,290 updates. All runs also logged spectral norms during training.
- **`qwen/`, `gemma/`:** all validation results, the training loss of every 100th update, and the last update of each segment. A run that we stopped and resumed has more than one segment, and the `segment` column gives its number. The elapsed time starts again in each segment, and the files keep repeated steps.

## Training time and MFU

The training time does not include validation. For Modded-NanoGPT, it is `train_time_ms` after the last update. For Qwen3 and Gemma 3, it is the elapsed time in each segment minus the validation time. MFU divides the FLOPs per second by the [dense BF16 peak of the B200](https://www.nvidia.com/en-us/data-center/hgx/), 2.25 PFLOP/s:

```math
\mathrm{MFU}=\frac{\text{FLOPs}}{\text{time}\times\text{peak FLOP/s}}.
```

One full run uses 2.413–2.418 × 10¹⁷ FLOPs for Modded-NanoGPT, 5.148–5.155 × 10¹⁹ FLOPs for Qwen3-0.6B, and 1.360–1.363 × 10²⁰ FLOPs for Gemma 3-1B. The value changes with the method.
