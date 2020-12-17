
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    general_group = parser.add_argument_group(title="general")
    general_group.add_argument("--dataset", type=str, default="mosei")
    general_group.add_argument("--do_trans", action="store_true", default=False)
    general_group.add_argument("--do_regre", action="store_true", default=False)
    general_group.add_argument("--batch_size", type=int, default=24)
    general_group.add_argument("--max_len", type=int, default=50)
    general_group.add_argument("--seed", type=int, default=123456)
    general_group.add_argument("--device_ids", type=int, nargs='+', default=0)
    general_group.add_argument("--train_info", type=str, default=None)

    regre_group = parser.add_argument_group(title="regression")
    regre_group.add_argument("--regre_save_path", type=str, default="./save/ccn/regre/")
    regre_group.add_argument("--regre_result_path", type=str, default="./result/ccn/regre/")
    regre_group.add_argument("--regre_epoch", type=int, default=20)
    regre_group.add_argument("--regre_lr", type=float, default=1e-4)
    regre_group.add_argument("--regre_weight_decay", type=float, default=0.0)
    regre_group.add_argument("--regre_mask_theta", type=float, default=0.1)
    regre_group.add_argument("--regre_mask_topk", type=int, default=10)
    regre_group.add_argument("--regre_mi_mask_type", type=str, default='theta')
    regre_group.add_argument("--regre_mp_mask_type", type=str, default='topk')

    regre_group.add_argument("--regre_mtl_weight", type=float, default=0.05)
    # model parameters
    regre_group.add_argument("--regre_hidden_size", type=int, default=256)
    regre_group.add_argument("--regre_num_layers", type=int, default=5)
    regre_group.add_argument("--regre_num_heads", type=int, default=5)
    regre_group.add_argument("--regre_dropout", type=float, default=0.1)

    trans_group = parser.add_argument_group(title="translation")
    trans_group.add_argument("--trans_save_path", type=str, default="./save/ccn/trans/")
    trans_group.add_argument("--trans_result_path", type=str, default="./result/ccn/trans/")
    trans_group.add_argument("--trans_from_trained", action="store_true", default=False)
    trans_group.add_argument("--trans_epoch", type=int, default=20)
    trans_group.add_argument("--trans_lr", type=float, default=1e-3)
    trans_group.add_argument("--trans_weight_decay", type=float, default=0.0)
    # model parameters
    trans_group.add_argument("--trans_hidden_size", type=int, default=256)
    trans_group.add_argument("--trans_num_layers", type=int, default=1)
    trans_group.add_argument("--trans_num_heads", type=int, default=8)
    trans_group.add_argument("--trans_encoder_dropout", type=float, default=0.1)
    trans_group.add_argument("--trans_decoder_dropout", type=float, default=0.1)

    return parser


def get_config_from_args(args):
    config = {"general": dict(),
              "regression": dict(),
              "translation": dict(),
              "w2v_model": dict(),
              "w2a_model": dict()
              }

    config["general"]["dataset"] = args.dataset
    config["general"]["batch_size"] = args.batch_size
    config["general"]["do_trans"] = args.do_trans
    config["general"]["do_regre"] = args.do_regre
    config["general"]["seed"] = args.seed
    config["general"]["device_ids"] = args.device_ids

    config["regression"]["save_path"] = args.regre_save_path
    config["regression"]["result_path"] = args.regre_result_path
    if not config["regression"]["save_path"].endswith('/'):
        config["regression"]["save_path"] += '/'
    if not config["regression"]["result_path"].endswith('/'):
        config["regression"]["result_path"] += '/'

    config["regression"]["max_len"] = args.max_len
    config["regression"]["w_size"] = 300
    config["regression"]["v_size"] = 35 if args.dataset == "mosei" else 47
    config["regression"]["a_size"] = 74
    config["regression"]["hidden_size"] = args.regre_hidden_size
    config["regression"]["num_layers"] = args.regre_num_layers
    config["regression"]["num_heads"] = args.regre_num_heads
    config["regression"]["head_size"] = args.regre_hidden_size // args.regre_num_heads
    config["regression"]["ff_size"] = args.regre_hidden_size // 2
    config["regression"]["dropout"] = args.regre_dropout
    config["regression"]["epoch"] = args.regre_epoch
    config["regression"]["lr"] = args.regre_lr
    config["regression"]["weight_decay"] = args.regre_weight_decay
    config["regression"]["mask_theta"] = args.regre_mask_theta
    config["regression"]["mask_topk"] = args.regre_mask_topk
    config["regression"]["mi_mask_type"] = args.regre_mi_mask_type
    config["regression"]["mp_mask_type"] = args.regre_mp_mask_type

    config["regression"]["mtl_weight"] = args.regre_mtl_weight

    config["translation"]["save_path"] = args.trans_save_path
    config["translation"]["result_path"] = args.trans_result_path
    if not config["translation"]["save_path"].endswith('/'):
        config["translation"]["save_path"] += '/'
    if not config["translation"]["result_path"].endswith('/'):
        config["translation"]["result_path"] += '/'

    config["translation"]["from_trained"] = args.trans_from_trained
    config["translation"]["epoch"] = args.trans_epoch
    config["translation"]["max_len"] = args.max_len
    config["translation"]["lr"] = args.trans_lr
    config["translation"]["weight_decay"] = args.trans_weight_decay

    config["w2v_model"]["max_len"] = args.max_len
    config["w2v_model"]["source_size"] = 300
    config["w2v_model"]["target_size"] = 35 if args.dataset == "mosei" else 47
    config["w2v_model"]["num_heads"] = args.trans_num_heads
    config["w2v_model"]["encoder_hidden_size"] = args.trans_hidden_size
    config["w2v_model"]["encoder_head_size"] = args.trans_hidden_size // args.trans_num_heads
    config["w2v_model"]["encoder_num_layers"] = args.trans_num_layers
    config["w2v_model"]["encoder_dropout"] = args.trans_encoder_dropout
    config["w2v_model"]["decoder_hidden_size"] = args.trans_hidden_size
    config["w2v_model"]["decoder_head_size"] = args.trans_hidden_size // args.trans_num_heads
    config["w2v_model"]["decoder_num_layers"] = args.trans_num_layers
    config["w2v_model"]["decoder_dropout"] = args.trans_decoder_dropout
    config["w2v_model"]["ff_size"] = args.trans_hidden_size // 2

    config["w2a_model"]["max_len"] = args.max_len
    config["w2a_model"]["source_size"] = 300
    config["w2a_model"]["target_size"] = 74
    config["w2a_model"]["num_heads"] = args.trans_num_heads
    config["w2a_model"]["encoder_hidden_size"] = args.trans_hidden_size
    config["w2a_model"]["encoder_head_size"] = args.trans_hidden_size // args.trans_num_heads
    config["w2a_model"]["encoder_num_layers"] = args.trans_num_layers
    config["w2a_model"]["encoder_dropout"] = args.trans_encoder_dropout
    config["w2a_model"]["decoder_hidden_size"] = args.trans_hidden_size
    config["w2a_model"]["decoder_head_size"] = args.trans_hidden_size // args.trans_num_heads
    config["w2a_model"]["decoder_num_layers"] = args.trans_num_layers
    config["w2a_model"]["decoder_dropout"] = args.trans_decoder_dropout
    config["w2a_model"]["ff_size"] = args.trans_hidden_size // 2

    return config


args = get_argparse().parse_args()
config = get_config_from_args(args)
