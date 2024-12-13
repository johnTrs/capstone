# OCR and KIE in FUNSD


#### OCR-> Text detection and recognition for words(azure) and text_lines(azure , surya): 
Parse dataset -> DONE

Surya-ocr -> DONE

Microsoft Azure OCR -> DONE

Evalueate OCR -> DONE ( maybe do F1 scores with the mapped words too instead of just naive and maybe WER , CER)

SER-> DONE (BIO tags with LayoutLM3base token classifier using word and segment level boxes, with hp tuning using optuna (3 models for std)

RE -> DONE (Bros w/ spade decoder finetune 100 epochs with open source code from github using word boxes)


MAYBE TODO:
BROS FOR RE (spade) using segment boxes ( 50 epochs) 

GeolayoutLM SER and RE ??? 

BROS for SER with spade, without spade ???
