# CUDA_VISIBLE_DEVICES=0 python eval.py --subset medical_flashcards --strategy instruct &
# CUDA_VISIBLE_DEVICES=1 python eval.py --subset medical_flashcards --strategy contrast &
# CUDA_VISIBLE_DEVICES=2 python eval.py --subset wikidoc --strategy instruct &
# CUDA_VISIBLE_DEVICES=3 python eval.py --subset wikidoc --strategy contrast &


: << 1
CUDA_VISIBLE_DEVICES=0 python eval.py --subset wikidoc --strategy original &
CUDA_VISIBLE_DEVICES=1 python eval.py --subset wikidoc --strategy mask &
CUDA_VISIBLE_DEVICES=2 python eval.py --subset medical_flashcards --strategy original &
CUDA_VISIBLE_DEVICES=3 python eval.py --subset medical_flashcards --strategy mask &
1

CUDA_VISIBLE_DEVICES=0 python eval.py --subset wikidoc --strategy remove &
CUDA_VISIBLE_DEVICES=1 python eval.py --subset wikidoc --strategy loss &
CUDA_VISIBLE_DEVICES=2 python eval.py --subset medical_flashcards --strategy remove &
CUDA_VISIBLE_DEVICES=3 python eval.py --subset medical_flashcards --strategy loss &


