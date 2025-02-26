# Custom Byaldi Extension

This module extends the base Byaldi library with custom functionality:

1. Overrides `RAGMultiModalModel` to use our custom `ColPaliModel` implementation
2. Adds Apple Silicon compatibility by forcing CPU usage on Apple ARM64 devices
3. Implements memory-mapped tensor loading for improved memory efficiency

## Installation

This module is designed to be installed as an editable package after the base Byaldi package:

```bash
pip install byaldi
pip install -e .
```

## Usage

Import the custom `RAGMultiModalModel` instead of the original:

```python
from custom_modules.byaldi import RAGMultiModalModel
```

The rest of the API remains the same as the base Byaldi library. 