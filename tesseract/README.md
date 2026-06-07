# **Tesseract.js Assets**

This directory contains the necessary assets for the OCR (Optical Character Recognition) engine used by **VDO Magnifier**.

### **Contents**

* **Engine Wrappers:** tesseract.min.js, worker.min.js.  
* **WebAssembly Cores:** tesseract-core.wasm.js, tesseract-core-simd.wasm.js.  
* **Language Data:** .traineddata.gz files for OCR model support.

### **Licensing & Credits**

These assets are distributed under the **Apache License 2.0**. All original copyright and license headers have been preserved within the source files as required.

This distribution includes bundled sub-dependencies, including:

* buffer (MIT)  
* ieee754 (BSD-3-Clause)  
* regenerator-runtime (MIT)  
* zlib.js (MIT)

Detailed license information for these sub-dependencies can be found in the provided \*.LICENSE.txt files.

### **Source**

These files were sourced from the official [Tesseract.js](https://github.com/naptha/tesseract.js) distribution via jsDelivr and Naptha's tessdata server. They are hosted locally in this repository to enable offline-capable functionality and to eliminate external runtime dependencies.