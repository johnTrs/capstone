### SOS All the models i will use should not be trained or fine tuned in the FUNSD dataset!!!!

In the papers i've red they use the ground truth FUNSD data to train the models and they dont perform their own ocr. Difirentiates us.

SOS: Some documents in the FUNSD have handwritten fields!!!! Use handwritten OCR?
### SOS Microsoft Azure OCR handles both printed and handwritten text SOS
the specific details about the internal architecture, including the
exact models and backbones used for recognition and detection tasks, are proprietary to
Microsoft and are not publicly disclosed in detail
__________________________________________________________________________________________________________________________________________________________________________________________________


Surya-ocr -> DONE
Parse dataset -> DONE
Microsoft Azure
Re run Florence
Handwritten charachters OCR????? -> azure
### Evalueate OCR's(???) and use the best OCR combined with ground truth for training ->>> SOS How to combine OCR with ground truth??????????????????????? SOS



Problems:

1) Florence misses the last tokens of the image sometimes in OCR with region. WTF? but florence 2 large-ft does it correct.  FLD-5B dataset is used for florence2
CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
error when running florence

2) PP_OCR doesnt run problem with gpu/setup

3) push to github 



__________________________________________________________________________________________________________________________________________________________________________________________________