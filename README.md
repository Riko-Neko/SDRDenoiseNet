**SDRDenoiseNet** is a deep learning-based system designed for denoising and signal detection in Software-Defined Radio (SDR) applications, with a focus on astrophysical signals such as neutral hydrogen (HI) emissions. Leveraging a custom convolutional neural network (FreqNet), the project processes noisy radio spectra to produce clean signals and generates probability matrices to identify signal peaks with high precision. Tailored for real-time processing on resource-constrained devices like Raspberry Pi, it supports both synthetic and real-world datasets (e.g., ALFALFA survey). The repository includes training scripts, inference pipelines, and visualization tools for analyzing denoising performance and signal detection accuracy.

Key features:
- **Advanced Denoising**: Utilizes a U-Net-inspired architecture with skip connections for superior noise reduction.
- **Signal Detection**: Outputs probability matrices to pinpoint HI signal peaks, optimized for sparse and weak signals.
- **SDR Compatibility**: Designed for SDR-based radio astronomy, with efficient processing for low-power hardware.
- **Comprehensive Visualization**: Includes plotting tools to compare noisy, clean, and denoised spectra with probability overlays.

Ideal for researchers and enthusiasts in radio astronomy, SDRDenoiseNet bridges deep learning and radio signal processing to enhance the detection of faint astrophysical signals in noisy environments.
