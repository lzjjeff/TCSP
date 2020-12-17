export CUDA_VISIBLE_DEVICES=0
python run_mimpt.py \
--dataset mosei \
--batch_size 24 \
--do_trans \
--trans_save_path ./save/trans/ \
--trans_result_path ./result/trans/ \
--device_ids 0 \
--trans_epoch 40 \
--trans_lr 1e-4 \
--trans_hidden_size 40 \
--trans_num_heads 5 \
--trans_num_layers 1