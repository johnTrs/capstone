# OCR and KIE in FUNSD

##### Microsoft Azure OCR handles both printed and handwritten text 
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
Evalueate OCR -> DONE ( maybe do F1 scores with the mapped words too instead of just naive and maybe WER , CER)
SER-> BIO tags with LayoutLM3base token classifier using word and segment level boxes, with hp tuning using optuna (3 models for std)
RE -> Bros w/ spade decoder finetune 100 epochs with open source code from github using word boxes


MAYBE TODO:
BROS FOR RE (spade) using segment boxes ( 50 epochs) 
GeolayoutLM SER and RE ??? 
BROS for SER with spade, without spade ???
