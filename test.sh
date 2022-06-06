nohup python -u ./agent.py --mode test --add_dueling --double --input image --gpu 0 --file_name ./image_out/ >> image.log 2>&1 &
nohup python -u ./agent.py --mode test --add_dueling --double --input vector --gpu 0 --file_name ./vector_out/ >> vector.log 2>&1 &
nohup python -u ./agent.py --mode test --add_dueling --double --input vector_attention --gpu 0 --file_name ./vector_attention_out/ >> vector_attention.log 2>&1 &
nohup python -u ./agent.py --mode test --add_dueling --double --input image_attention --gpu 1 --file_name ./image_attention_out/ >> image_attention.log 2>&1 &
nohup python -u ./agent.py --mode test --add_dueling --double --input multi_attention --gpu 1 --file_name ./multi_attention_out/ >> multi_attention.log 2>&1 &