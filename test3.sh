nohup python -u ./agent.py --mode test --add_dueling --input multi_attention --gpu 0 --file_name ./result3/duelingdqn_out/ >> ./result3/duelingdqn.log 2>&1 &

nohup python -u ./agent.py --mode test --add_dueling --double --input multi_attention --right_r 0.4 --gpu 0 --file_name ./result2/incre_r_out/ >> ./result2/incre_r.log 2>&1 &
nohup python -u ./agent.py --mode test --double --input multi_attention --gpu 1 --file_name ./result3/ddqn_out/ >> ./result3/ddqn.log 2>&1 &
nohup python -u ./agent.py --mode test --input multi_attention --gpu 1 --file_name ./result3/dqn/ >> ./result3/dqn.log 2>&1 &
