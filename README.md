# CT-XRay-Diagnostics

## Overview

**CT-XRay-Diagnostics** is an AI-driven application designed to analyze X-ray and CT scan images for enhanced disease detection and diagnostics in the medical field. By leveraging advanced deep learning models, this tool provides healthcare professionals with immediate analysis and insights into various medical conditions.

## Features

- **Image Upload:** Users can easily upload X-ray or CT scan images for analysis.
- **Disease Prediction:** The application identifies potential diseases, including pneumonia, tuberculosis, and other thoracic conditions.
- **Detailed Explanations:** Along with predictions, users receive explanations about the identified conditions, aiding in better understanding and decision-making.

## Technology Stack

- **Models Used:** The application utilizes Vision Transformers (ViTs) for high-performance image classification.
- **Framework:** Built with Streamlit for a user-friendly interface.

## Getting Started

### Prerequisites

- Python 3.10.7
- Required libraries (can be installed via `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:haroon423/CT-XRay-Diagnostics.git
   cd CT-XRay-Diagnostics

pip install -r requirements.txt

streamlit run app.py

