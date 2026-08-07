 HEAD
\documentclass[11pt]{article}

\usepackage{hyperref}
\usepackage{listings}
\usepackage{geometry}
\geometry{margin=1in}

\title{DAM-PARSeq: Deformable Attention Module Enhanced PARSeq for Robust Scene Text Recognition}

\author{}

\begin{document}

\maketitle


\section{Overview}

This repository provides the implementation of \textbf{DAM-PARSeq}, a modified version of the original PARSeq framework for robust Scene Text Recognition (STR).

The original PARSeq framework achieves strong recognition performance by combining a Vision Transformer (ViT) encoder with a Permutation Language Modeling (PLM)-based decoder. However, visual degradation factors including blur, occlusion, distortion, illumination variation, and background interference can reduce the quality of extracted visual representations.

To address these limitations, DAM-PARSeq introduces a \textbf{Deformable Attention Module (DAM)} between the ViT encoder and the PARSeq decoder. DAM adaptively refines visual tokens before sequence prediction while preserving the original PARSeq decoding strategy.


\section{Method Overview}

\subsection{Baseline-PARSeq}

The original PARSeq framework follows:

\begin{verbatim}
Input Image
      |
      v
ViT Encoder
      |
      v
Visual Tokens
      |
      v
PARSeq Decoder
      |
      v
Output Sequence
\end{verbatim}

The visual features extracted from the encoder are directly transferred to the decoder.


\subsection{Proposed DAM-PARSeq}

The proposed framework inserts DAM between the ViT encoder and PARSeq decoder:

\begin{verbatim}
Input Image
      |
      v
ViT Encoder
      |
      v
Visual Tokens
      |
      v
Deformable Attention Module (DAM)
      |
      v
Enhanced Visual Features
      |
      v
PARSeq Decoder
      |
      v
Output Sequence
\end{verbatim}

DAM learns adaptive sampling locations and attention weights to enhance informative visual regions while reducing degraded visual responses.

The original PARSeq decoder and permutation language modeling strategy remain unchanged.


\section{Repository Structure}

\begin{verbatim}
DAM-PARSeq/

|-- configs/
|
|-- datasets/
|
|-- models/parseq
|          |-- Baseline-Parseq/
|          |-- DAM-Parseq/
|
|-- train.py
|-- test.py
|-- requirements.txt
|-- README.tex

\end{verbatim}


\section{Getting Started}

This repository is developed based on the original PARSeq implementation:

\url{https://github.com/baudm/parseq}

The original training pipeline, dataset processing procedures, and evaluation protocols are maintained. The main modification is the integration of DAM after the ViT encoder.


\section{Installation}

Requirements:

\begin{itemize}
\item Python $\geq$ 3.9
\item PyTorch $\geq$ 2.0
\end{itemize}


Create environment:

\begin{verbatim}
conda create -n DAM_parseq python=3.9
conda activate DAM_parseq
\end{verbatim}


Install dependencies:

\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}


\section{Dataset Preparation}

DAM-PARSeq follows the original PARSeq evaluation protocol.

\subsection{Training Datasets}

Synthetic training datasets:

\begin{itemize}
      \item Training datasets: MJ + ST
      \item Same PARSeq optimization pipeline
\end{itemize}
The synthetic training datasets (MJ and ST) and benchmark evaluation datasets follow the original 
PARSeq dataset preparation protocol.: https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE

\subsection{Benchmark Evaluation Datasets}

Standard scene text recognition benchmarks:

\begin{itemize}
\item IIIT5K
\item SVT
\item IC13
\item IC15
\item SVTP
\item CUTE80
\end{itemize}

Dataset preparation follows the original PARSeq repository.


\section{Model Modification}

The original PARSeq framework:

\begin{verbatim}
visual_features = encoder(image)

prediction = decoder(visual_features)
\end{verbatim}


DAM-PARSeq:

\begin{verbatim}
visual_features = encoder(image)

refined_features = DAM(visual_features)

prediction = decoder(refined_features)
\end{verbatim}


DAM improves visual representation quality before decoding without modifying the PARSeq decoder architecture.


\section{Training Procedure}

The training procedure follows the original PARSeq framework.

Training steps:

\begin{enumerate}
\item Input images are processed by the ViT encoder.
\item Visual tokens are extracted.
\item DAM refines visual representations.
\item Enhanced visual features are passed to the PARSeq decoder.
\item Recognition loss is optimized.
\end{enumerate}


Example training command:

\begin{verbatim}
python train.py
\end{verbatim}


\section{Evaluation Procedure}

Evaluation can be performed using:

\begin{verbatim}
python test.py <checkpoint_path>
\end{verbatim}


Evaluation metrics include:

\begin{itemize}
\item Accuracy
\item 1-Normalized Edit Distance (1-NED)
\item Confidence Score
\end{itemize}


\section{Real Printed-Sign Dataset}

DAM-PARSeq was additionally evaluated on a real printed-sign dataset containing 804 images.

The dataset contains challenging degradation conditions:

\begin{itemize}
\item Motion blur
\item Occlusion
\item Distortion
\item Low illumination
\item Background interference
\item Partially visible characters
\end{itemize}

The dataset was collected by the authors for robustness evaluation and is not publicly available due to privacy considerations.


\section{Reproducing DAM-PARSeq Results}

To reproduce the experimental results:

\begin{enumerate}
\item Prepare the synthetic STR datasets.
\item Install all required dependencies.
\item Train DAM-PARSeq using the provided training pipeline.
\item Evaluate the trained model using the testing script.
\item Compare performance using Accuracy, 1-NED, and Confidence metrics.
\end{enumerate}


\section{Citation}

This repository is based on the original PARSeq implementation. If you use this repository, please cite the PARSeq paper:

\begin{verbatim}
@InProceedings{bautista2022parseq,
title={Scene Text Recognition with 
Permuted Autoregressive Sequence Models},
author={Bautista, Darwin and Atienza, Rowel},
booktitle={European Conference on Computer Vision},
pages={178--196},
year={2022},
publisher={Springer Nature Switzerland}
}
\end{verbatim}


\end{document}
=======
# DAM-PARSeq
Robustness-Enhanced STR for Occluded and Blurred Printed Signs Using a Deformable Attention Module53ce432635722773106ddd123a6f413a09c7f15d
