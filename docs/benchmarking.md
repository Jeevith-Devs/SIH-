# ASTRA-Track Benchmarking & Performance Evaluation

## 1. Official SIH 2026 Problem Statement 26169 Target Criteria

| Metric Parameter | SIH Target Benchmark | Mathematical Definition |
| :--- | :--- | :--- |
| **Acquisition Time** | $\le 2.0\text{ s}$ | $t_{\text{lock}} - t_{\text{start}}$ |
| **Tracking Error** | $\le 10.0\text{ px}$ | $\sqrt{(x_{\text{actual}} - x_{\text{tracked}})^2 + (y_{\text{actual}} - y_{\text{tracked}})^2}$ |
| **Target Loss Rate** | $< 5.0\%$ | $\frac{N_{\text{lost}}}{N_{\text{total}}} \times 100\%$ |
| **Reacquisition Time** | $\le 1.0\text{ s}$ | $t_{\text{restored}} - t_{\text{loss\_start}}$ |
| **Processing Throughput** | $\ge 20.0\text{ FPS}$ | $\frac{1}{\Delta t_{\text{frame}}}$ |
| **Continuous Lock Retention** | $\ge 95.0\%$ | $\frac{N_{\text{locked}}}{N_{\text{total}}} \times 100\%$ |

## 2. Automated Batch Testing
Users can run automated test batteries over 5, 10, 50, or 100 scenarios across diverse trajectory types, atmospheric conditions, and jitter amplitudes:
```bash
python main.py --benchmark --count 5
```

## 3. Multi-Variable Stress Test
Randomly samples beacon speeds ($50\text{--}110\text{ px/s}$), trajectory geometries, fog intensities ($10\text{--}65\%$), salt-and-pepper noise ($0\text{--}12\%$), and platform vibration amplitudes ($0\text{--}14\text{ px}$):
```bash
python main.py --stress-test --count 10
```
Reports Best-Case, Average-Case, Worst-Case metrics, and overall Pass Rate.
