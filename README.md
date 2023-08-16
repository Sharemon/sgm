# sgm
cpu&gpu version of sgm algorithm for stereo matching.

---

compared to the cpu version, gpu version has 50 times speed up.

| image resolution | time cost of cpu version | time cost of gpu version |
| ---- | ---- | ---- |
| 472 x 376 | 522.4ms @ single core of Intel E3-1231 | 10.14ms @ GTX1060 |

# References
* Heiko Hirschmüller. Hirschmüller, H: Stereo processing by semiglobal matching and mutual information. IEEE PAMI 30(2), 328-341[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008, 30(2):328-341.
* D. Hernandez-Juarez, A. Chacón, A. Espinosa, D. Vázquez, J. C. Moure, and A. M. López. Embedded real-time stereo estimation via Semi-Global Matching on the GPU.  ICCS2016 – International Conference on Computational Science 2016
