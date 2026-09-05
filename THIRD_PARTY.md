# Scientific and software attribution

The BRF equations and reproduction recipes derive from Higuchi et al. (2024)
and their reference implementation at AdaptiveAILab/brf-neurons, commit
`1a42b8c8aceedb13cae3b2327774c2fcc04fd696`. This package is an independent
implementation. The authors' repository is used as an external numerical oracle
and as the source of the published ECG preprocessing. The following notice is
retained for any portions derived from that software:

MIT License

Copyright (c) 2024 Adaptive AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SHD and SSC: Benjamin Cramer, Yannik Stradmann, Johannes Schemmel and Friedemann
Zenke, *The Heidelberg Spiking Data Sets for the Systematic Evaluation of
Spiking Neural Networks*, IEEE TNNLS 33, 2744–2757 (2022),
doi:10.1109/TNNLS.2020.3044364. Data licensed CC BY 4.0. This repository
downloads from the authors' server and does not redistribute their data.

MNIST: Yann LeCun, Corinna Cortes and Christopher J. C. Burges. Downloaded from
the public OSSCI mirror without changing the supplied training/test split.

QTDB: the preprocessed version distributed by Yin et al. and subsequently BRF;
consult PhysioNet's QT Database and the originating preprocessing repository
before any dataset redistribution. Only download URLs and digests are published.
