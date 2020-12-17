export CUDA_VISIBLE_DEVICES=0
#python run_transformer.py \
#--dataset mosei \
#--batch_size 24 \
#--do_trans \
#--save_path ./transformer/save \
#--result_path ./transformer/result/test/ \
#--device_ids 0 \
#--trans_epoch 20 \
#--trans_lr 1e-4 \
#--trans_hidden_size 80 \
#--trans_n_heads 5 \
#--trans_n_layers 5 \
#--regre_epoch 20 \
#--regre_lr 1e-3 \
#--regre_hidden_size 40 \
#--regre_n_heads 5 \
#--regre_n_layers 5
#
python run_transformer.py \
--dataset mosei \
--batch_size 24 \
--do_regre \
--save_path ./transformer/save \
--result_path ./transformer/result/ \
--device_ids 0 \
--fixed \
--use_shared_list \
--shared_list 0N1N2 \
--trans_epoch 20 \
--trans_lr 1e-3 \
--trans_hidden_size 80 \
--trans_n_heads 5 \
--trans_n_layers 5 \
--regre_epoch 20 \
--regre_lr 1e-3 \
--regre_hidden_size 40 \
--regre_n_heads 5 \
--regre_n_layers 5
#
#python run_transformer.py \
#--dataset mosei \
#--batch_size 24 \
#--do_regre \
#--result_path ./transformer/result \
#--device_ids 0 \
#--trans_epoch 20 \
#--trans_lr 1e-3 \
#--trans_hidden_size 80 \
#--trans_n_heads 5 \
#--trans_n_layers 5 \
#--regre_epoch 20 \
#--regre_lr 1e-3 \
#--regre_hidden_size 40 \
#--regre_n_heads 5 \
#--regre_n_layers 5