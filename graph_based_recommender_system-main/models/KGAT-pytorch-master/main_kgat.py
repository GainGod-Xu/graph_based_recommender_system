import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT

import mlflow
from mlflow import log_metric, log_param, log_artifacts
import optuna

mlflow.set_tracking_uri("http://52.53.202.9:5000/")
mlflow.set_experiment("kgat_amzn_focal_loss_test")


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    metrics_dict['recip_rank'] = list()

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            metrics_dict['recip_rank'].append(batch_metrics['recip_rank'])
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    metrics_dict['mrr'] = np.concatenate(metrics_dict['recip_rank']).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            # logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
            #     epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[1]['precision'], metrics_dict[3]['precision'], metrics_dict[5]['precision'], metrics_dict[10]['precision'], metrics_dict[100]['precision'],
                metrics_dict[1]['recall'], metrics_dict[3]['recall'], metrics_dict[5]['recall'], metrics_dict[10]['recall'], metrics_dict[100]['recall'],
                metrics_dict[1]['ndcg'], metrics_dict[3]['ndcg'], metrics_dict[5]['ndcg'], metrics_dict[10]['ndcg'], metrics_dict[100]['ndcg']))
            
            for k in [1, 3, 5, 10, 100]:
                for metric in ['recall', 'precision', 'ndcg']:
                    log_metric('valid_'+str(metric)+'_at_'+str(k), float(metrics_dict[k][metric]), step=epoch)
            
            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                model_path = save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    # logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
    #     int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@1'], best_metrics['precision@3'], best_metrics['precision@5'], best_metrics['precision@10'], best_metrics['precision@100'],
        best_metrics['recall@1'], best_metrics['recall@3'], best_metrics['recall@5'], best_metrics['recall@10'], best_metrics['recall@100'],
        best_metrics['ndcg@1'], best_metrics['ndcg@3'], best_metrics['ndcg@5'], best_metrics['ndcg@10'], best_metrics['ndcg@100']))
    return model_path, best_recall


def predict(args, model_path, test_file='test.txt'):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    args.test_file = test_file
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    
    logging.info('CF Evaluation: Precision [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
        metrics_dict[1]['precision'], metrics_dict[3]['precision'], metrics_dict[5]['precision'], metrics_dict[10]['precision'], metrics_dict[100]['precision'],
        metrics_dict[1]['recall'], metrics_dict[3]['recall'], metrics_dict[5]['recall'], metrics_dict[10]['recall'], metrics_dict[100]['recall'],
        metrics_dict[1]['ndcg'], metrics_dict[3]['ndcg'], metrics_dict[5]['ndcg'], metrics_dict[10]['ndcg'], metrics_dict[100]['ndcg']))

    for k in [1, 3, 5, 10, 100]:
        for metric in ['recall', 'precision', 'ndcg']:
            log_metric('test_'+str(metric)+'_at_'+str(k), float(metrics_dict[k][metric]))
    




if __name__ == '__main__':
    global STATIC_PARAMS
    STATIC_PARAMS = {
        'use_focal' : 0.9999, 
        'n_epoch' : 1000,
        'Ks' : '[1,3,5,10,100]',
        'evaluate_every' : 1,
        'stopping_steps' : 10,
        'cf_print_every' : 100,
        'kg_print_every' : 100,
        'data_name' : 'amazon-product-review/kgat_data',
        'data_dir' : 'data/',
        'use_pretrain' : 0,
        'test_batch_size' : 10000,
        'conv_dim_list' : '[64, 32, 16]',
        'mess_dropout' : '[0.1, 0.1, 0.1]',
        'relation_dim' : 64,
	}

    def objective(trial):
        params = {
            'cf_batch_size' : trial.suggest_int('cf_batch_size', 1024, 1024),
            'kg_batch_size' : trial.suggest_int('kg_batch_size', 1024, 2048),
            'embed_dim' : trial.suggest_int('embed_dim', 200, 200),
            'laplacian_type' : trial.suggest_categorical('laplacian_type', ['random-walk', 'symmetric']),
            'aggregation_type' : trial.suggest_categorical('aggregation_type', ['bi-interaction', 'gcn', 'graphsage']),
            'kg_l2loss_lambda' : trial.suggest_float('kg_l2loss_lambda', 1e-5, 1e-2),
            'cf_l2loss_lambda' : trial.suggest_float('cf_l2loss_lambda', 1e-5, 1e-2),
            'lr' : trial.suggest_float('lr', 0.0001, 0.01),
        }
        final_params = STATIC_PARAMS | params
        mlflow.start_run()
        for param in final_params:
            log_param(param, final_params[param])

        hyper_params = []
        for param in final_params:
            hyper_params.append('--'+param)
            hyper_params.append(str(final_params[param]))
        score = fit(hyper_params)
        mlflow.end_run()
        return score

    def fit(hyper_params):
        
        
        args = parse_kgat_args(hyper_params)
        model_path, valid_score = train(args)
        predict(args, model_path)
        return valid_score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, catch=(RuntimeError,))
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image("kgat_amazn_tune_focal.png")

