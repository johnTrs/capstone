##### SOS All the models i will use should not be trained or fine tuned in the FUNSD dataset!!!!

In the papers i've red they use the ground truth FUNSD data to train the models and they dont perform their own ocr. Difirentiates us.
We use end-end from OCR->KIE, we only use the annotation of the tokens which can be implemented by 
a company using a tool.

SOS: Some documents in the FUNSD have handwritten fields!!!! 
##### SOS Microsoft Azure OCR handles both printed and handwritten text SOS
Paper : Synergizing Optical Character Recognition: A
Comparative Analysis and Integration of Tesseract,
Keras, Paddle, and Azure OCR
σελιδα 22
the specific details about the internal architecture, including the
exact models and backbones used for recognition and detection tasks, are proprietary to
Microsoft and are not publicly disclosed in detail
__________________________________________________________________________________________________________________________________________________________________________________________________


#### OCR extract text and text location for words(azure) and text_lines(surya,florence): 
Parse dataset -> DONE
Surya-ocr -> DONE
Microsoft Azure OCR -> DONE
Re run Florence
Handwritten charachters OCR????? -> azure how is it performing there?
##### Evalueate OCR's(???) and use the best OCR combined with ground truth for training ->>> SOS How to combine OCR with ground truth??????????????????????? SOS
##### We annotated ner tags and linking by matching our ocr to the ground truth, a company can do that to its ocr tokens using some tool or by collecting Large 




Problems:
1) Florence misses the last tokens of the image sometimes in OCR with region. WTF? but florence 2 large-ft does it correct.  FLD-5B dataset is used for florence2
CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
error when running florence

2) OCR evaluation 

3) Layout lm hyperparameter tuning / Validation set?? hot to split?




__________________________________________________________________________________________________________________________________________________________________________________________________