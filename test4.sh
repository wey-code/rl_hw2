nohup python -u ./agent.py --mode test --model_ep 800 --isdeadnot --add_dueling --double --input multi_attention  --gpu 0 --file_name ./result4/remove_dead/ >> ./result4/remove_dead.log 2>&1 &

nohup python -u ./agent.py --mode test --model_ep 800 --add_dueling --double --input multi_attention  --gpu 1 --file_name ./result4/normal/ >> ./result4/normal.log 2>&1 &