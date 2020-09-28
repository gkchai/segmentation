# author: kcgarikipati@gmail.com

"""main script"""
import model
import loader
import tensorflow as tf
import os
import sys
from tqdm import tqdm
import shutil
import settings
import time
from utils.settings import SettingsManager
from utils.general import dict_to_obj, dataset_to_path, merge_dicts,\
    n_in_csv, get_logger, silence_tf_logger, make_dir
import json
from datetime import datetime
import math
from collections import deque
import pdb
import traceback

# silence TF messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
silence_tf_logger()
logger = get_logger('main')


class Segmentation:

    def __init__(self, cfg, info):
        self.cfg = cfg
        self.info = info
        # set seed before calling loader ?
        tf.set_random_seed(cfg.seed)
        self.dataloader = loader.Loader(cfg)
        self.model = model.Model(cfg)
        self.update_csv_path()
        self.make_paths()
        self.make_dirs()
        self.n_train, self.n_val, self.n_test = self.get_data_info()
        # logger.info("Using the following configuration: {}".format(self.cfg.__dict__))

    def update_csv_path(self):
        self.train_csv = dataset_to_path(self.cfg.train_data,
                                       os.path.join(self.cfg.data_dir, 'train'), self.cfg.dim)
        self.val_csv = dataset_to_path(self.cfg.val_data,
                                      os.path.join(self.cfg.data_dir, 'val'), self.cfg.dim)
        self.test_csv = dataset_to_path(self.cfg.test_data,
                                      os.path.join(self.cfg.data_dir, 'test'), self.cfg.dim)

    def make_paths(self):
        self.ck_path = os.path.join(self.cfg.ck_dir, self.cfg.model_id)
        self.train_log_path = os.path.join(self.cfg.log_dir, self.cfg.model_id, 'train')
        self.val_log_path = os.path.join(self.cfg.log_dir, self.cfg.model_id, 'val')
        self.test_log_path = os.path.join(self.cfg.log_dir, self.cfg.model_id, 'test')
        self.exp_path = os.path.join(self.cfg.export_dir, self.cfg.model_id)
        self.out_path = os.path.join(self.cfg.out_dir, self.cfg.model_id)
        self.info_path = os.path.join(self.ck_path, 'info.json')

    def make_dirs(self):
        # base dirs
        make_dir(self.cfg.ck_dir)
        make_dir(self.cfg.log_dir)
        make_dir(os.path.join(self.cfg.log_dir, self.cfg.model_id))
        make_dir(self.cfg.export_dir)
        make_dir(self.cfg.out_dir)

        # model dirs
        make_dir(self.ck_path)
        make_dir(self.exp_path)
        make_dir(self.train_log_path)
        make_dir(self.val_log_path)
        make_dir(self.test_log_path)

    @staticmethod
    def run_epoch(epoch_no, is_train, feed_dict_fn, n_batches, sess, data_op, run_op, summary_writer):
        # perform one complete epoch
        dice_loss, sce_loss, jacc_loss, conf, count = 0, 0, 0, 0, 0
        for _ in tqdm(list(range(0, n_batches)), desc='batch'):

            X_batch, w_batch, y_batch = sess.run(data_op)
            # pdb.set_trace()
            feed_dict = feed_dict_fn(X_batch, w_batch, y_batch, is_train)
            _, loss, loss_d, loss_p, loss_j, step_summary, global_step_value, _, _, step_conf = \
                sess.run(run_op, feed_dict=feed_dict)

            dice_loss += loss_d
            sce_loss += loss_p
            jacc_loss += loss_j
            conf += step_conf
            count += 1

            if summary_writer:
                summary_writer.add_summary(step_summary, global_step_value)
                epoch_summary = tf.Summary()
                epoch_summary.value.add(tag='epoch_losses/dice_loss', simple_value=dice_loss/count)
                epoch_summary.value.add(tag='epoch_losses/iou_loss', simple_value=jacc_loss/count)
                epoch_summary.value.add(tag='epoch_losses/sce_loss', simple_value=sce_loss/count)
                summary_writer.add_summary(epoch_summary, global_step=epoch_no + 1)

        metric_dict = {
            "dice": round(-dice_loss / count, 5),
            "jacc": round(-jacc_loss / count, 5),
            "sce": round(-sce_loss / count, 5),
            # "conf": round(conf/count, 5)
        }
        logger.info("Op: {0}, Dice coeff: {1:.4f}, Cross entropy : {2:.4f}, "
                    "Jacc coeff : {3:.4f}, Confidence : {4:.4f}".
                    format(('Train' if is_train else 'Val'), -dice_loss / count,
                           sce_loss / count, -jacc_loss / count, conf / count))
        return metric_dict

    @staticmethod
    def restore_checkpoint(sess, saver, ckpt):
        """if that checkpoint exists, restore from checkpoint"""
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Restoring checkpoint {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            logger.info("Checkpoint not found")
            return False

    def get_data_info(self):
        """get dataset count and also update metrics"""
        n_train, n_each_train = n_in_csv(self.train_csv)
        n_val, n_each_val = n_in_csv(self.val_csv)
        n_test, n_each_test = n_in_csv(self.test_csv)
        self.info['n_train'] = n_train
        self.info['n_val'] = n_val
        self.info['n_test'] = n_test
        self.info['metrics']['train_data'] = dict(zip(self.cfg.train_data, n_each_train))
        self.info['metrics']['val_data'] = dict(zip(self.cfg.test_data, n_each_val))
        self.info['metrics']['test_data'] = dict(zip(self.cfg.test_data, n_each_test))
        return n_train, n_val, n_test

    def write_info(self):
        """write the info dict to json file. For consistency, write only after each epoch when all metrics are
        available"""
        info_json = json.dumps(self.info, indent=4, sort_keys=True)
        with open(self.info_path,'w') as file:
            file.write(info_json)

    def train(self):
        # train and validate model

        logger.info("Training Model_id: {}".format(self.cfg.model_id))
        logger.info("Training dataset contains {} images".format(self.n_train))
        logger.info("Validation dataset contains {} images".format(self.n_val))
        logger.info("Test dataset contains {} images".format(self.n_test))

        tf.reset_default_graph()

        # generate data op
        train_data_op = self.dataloader.create_train_data_ops(self.train_csv)
        val_data_op = self.dataloader.create_val_data_ops(self.val_csv)

        loss_op, train_op, eval_op, summary_op, feed_dict_fn = self.model.build(is_train=True)

        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 1})) as sess:
            train_summary_writer = tf.summary.FileWriter(self.train_log_path, sess.graph)
            val_summary_writer = tf.summary.FileWriter(self.val_log_path)

            ckpt = tf.train.get_checkpoint_state(self.ck_path, latest_filename='checkpoint')
            best_ckpt = tf.train.get_checkpoint_state(self.ck_path, latest_filename='best_checkpoint')
            saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)

            # if could not restore checkpoint
            if not (self.restore_checkpoint(sess, saver, ckpt)):
                best_ckpt = None
                # tf.global_variables_initializer() is a time-saver but technically you do not have to call it
                # and could initialize your variables by other means (restoring weights from file).
                sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            save_path = os.path.join(self.ck_path, self.cfg.model_basename + '.ckpt')
            save_best_path = os.path.join(self.ck_path, self.cfg.best_model_basename + '.ckpt')
            best_saver = tf.train.Saver(max_to_keep=1)
            tf.train.write_graph(sess.graph_def, self.exp_path, self.cfg.model_basename + '.pbtxt')

            try:
                global_step = tf.train.get_global_step(sess.graph)

                if best_ckpt:
                    # we dont restore best checkpoint but the last one
                    info_dict = json.loads(open(self.info_path).read())
                    best_score = info_dict['metrics']['val_scores'][self.cfg.loss]
                    logger.info("Previous best {0} score = {1:.4f}".format(self.cfg.loss, best_score))
                    start_epoch = info_dict['curr_epoch']
                else:
                    best_score = -sys.maxsize
                    start_epoch = 0

                patience = 0
                curr_score_win = deque(maxlen=self.cfg.moving_window) # FIFO queue
                for epoch in range(start_epoch,  self.cfg.nb_epochs):

                    # update curr epoch count
                    self.info['curr_epoch'] = epoch+1

                    train_run_op = [train_op] + loss_op + [summary_op, global_step] + eval_op
                    val_run_op = [tf.no_op()] + loss_op + [summary_op, global_step] + eval_op

                    n_train_batches = math.ceil(self.n_train/self.cfg.batch_size)
                    train_metric = self.run_epoch(epoch, True, feed_dict_fn, n_train_batches, sess, train_data_op,
                             train_run_op, train_summary_writer)

                    n_val_batches = math.ceil(self.n_val/self.cfg.batch_size)
                    val_metric = self.run_epoch(epoch, False, feed_dict_fn, n_val_batches, sess, val_data_op,
                                                      val_run_op, val_summary_writer)

                    # append to curr_score, automatically drops old elements to keep fixed size
                    curr_score_win.append(val_metric[self.cfg.loss])

                    if (epoch + 1) % self.cfg.save_period == 0:
                        ret = saver.save(sess, save_path)
                        logger.info("Created {}".format(ret))
                        self.write_info()

                    # calculate moving average score (different from epoch score)
                    curr_score = sum(curr_score_win)/len(curr_score_win)
                    if curr_score > best_score + self.cfg.min_delta:
                        # save the ckpt with the best score
                        logger.info("Found new best {0} score = {1:.4f} > {2:.4f}".format(self.cfg.loss, curr_score,
                                                                                          best_score))
                        ret = best_saver.save(sess, save_best_path, latest_filename='best_checkpoint')
                        logger.info("Created {}".format(ret))
                        best_score = curr_score

                        # update the info metrics dict with current epoch scores
                        self.info['metrics']['train_scores'] = train_metric
                        self.info['metrics']['val_scores'] = val_metric
                        self.info['best_epoch'] = epoch + 1
                        self.write_info()
                        patience = 0
                    else:
                        patience += 1
                        if patience > self.cfg.patience:
                            logger.info("Score did not increase for past {0} epochs=>Early stopping".format(
                                patience))
                            break

                    logger.info("Completed epochs: {}".format(epoch + 1))
            finally:
                coord.request_stop()
                coord.join(threads)

    def test(self, input_csv=None):
        # test against each sub-dataset to get scores

        test_csv = dataset_to_path(input_csv, os.path.join(self.cfg.data_dir, 'test'), self.cfg.dim) \
            if input_csv else self.test_csv

        # test score for each dataset is stored in dict as well
        self.info['metrics']['each_test_scores'] = {}
        # retain stored epoch variables
        info_dict = json.loads(open(self.info_path).read())
        self.info['nb_epochs'] = info_dict['nb_epochs']
        self.info['best_epoch'] = info_dict['best_epoch']
        self.info['curr_epoch'] = info_dict['curr_epoch']

        test_tup = [(self.dataloader.create_test_data_ops([test_csv_e]), n_in_csv([test_csv_e])[0])
                                    for test_csv_e in test_csv]
        loss_op, _, eval_op, summary_op, feed_dict_fn = self.model.build(is_train=False)

        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 1})) as sess:
            ckpt = tf.train.get_checkpoint_state(self.ck_path, latest_filename='best_checkpoint')
            saver = tf.train.Saver()

            # if that checkpoint exists, restore from checkpoint
            if not self.restore_checkpoint(sess, saver, ckpt):
                sys.exit(0)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for test_tup_e, dataset in zip(test_tup, self.cfg.test_data):

                    test_data_op = test_tup_e[0]
                    n_test = test_tup_e[1]
                    logger.info("Test dataset {} contains {} images".format(dataset, n_test))

                    test_run_op = [tf.no_op()] + loss_op + [summary_op, tf.no_op()] + eval_op
                    n_batches = math.ceil(n_test/self.cfg.batch_size)

                    metrics_dict = self.run_epoch(0, False, feed_dict_fn, n_batches, sess, test_data_op,
                                                  test_run_op, None)
                    self.info['metrics']['each_test_scores'][dataset] = metrics_dict

                self.write_info()
            finally:
                coord.request_stop()
                coord.join(threads)

    def search(self):
        """grid search over given hyperparams"""
        pass


def main(argv):

    base_cfg = settings.Config()
    sm = SettingsManager.get(base_cfg.get_options(), base_cfg.__dict__)
    args_dict = sm.parse_cmd()
    sm.update(args_dict)

    # handle all the possible cases to start/resume/restart training
    if not args_dict['model_id']:
        model_id = datetime.now().strftime("%Y%m%d") + '.' + sm.get_hash(select_keys=base_cfg.get_hash_keys())
        sm.update({'model_id': model_id})
        ckpt_dir = os.path.join(base_cfg.ck_dir, model_id)

        if os.path.exists(ckpt_dir):
            # restart training
            if args_dict['restart']:
                logger.info("Delete old checkpoint dir {} and restart from scratch".format(ckpt_dir))
                rm_dirs = [ckpt_dir, os.path.join(base_cfg.log_dir, model_id)]
                for rm_dir in rm_dirs:
                    try:
                        shutil.rmtree(rm_dir)
                    except OSError as err:
                        logger.info("OS error: {0}".format(err))
            else:
                # special case: happens when you run training with same default settings within same day.
                # Hash gives same model_id, so just resume what you had
                logger.info("Checkpoint dir {} already exists for model {}. Will try to resume if checkpoint exists"
                            .format(ckpt_dir, model_id))

        else:
            # start new training
            logger.info("Start new training in checkpoint dir {}".format(ckpt_dir))
    else:
        model_id = args_dict['model_id']
        ckpt_dir = os.path.join(base_cfg.ck_dir, model_id)
        if not os.path.exists(ckpt_dir):
            logger.info("Provided model {} does not exist. Can't resume/restart something that doesn't exist".format(
                model_id))
            sys.exit(0)

        # resume training and ignore command line settings
        logger.info("Resume from old checkpoint dir {} with stored parameters".format(ckpt_dir))
        json_dict = json.loads(open(os.path.join(ckpt_dir, 'info.json')).read())
        stored_dict = merge_dicts(json_dict['params'], json_dict['dataset'])
        # given_dict = sm.__dict__()
        # # checking if cmd parameters are contained in stored parameters
        # if not dict_in_another(stored_dict, given_dict):
        #     logger.info("Stored settings don't match the command line settings. Using stored settings")
        sm.update(stored_dict)

    # create cfg class
    cfg = dict_to_obj(sm.__dict__())

    # create info dict
    info = sm.get_dict(base_cfg.get_common_keys())
    info.update({
            'params': sm.get_dict(base_cfg.get_params_keys()),
            'dataset': sm.get_dict(base_cfg.get_dataset_keys()),
            'metrics': {},
            }
    )
    cfg.height = cfg.dim
    cfg.width = cfg.dim

    seg = Segmentation(cfg, info)

    if cfg.phase == 'train':
        try:
            seg.train()
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

    if cfg.phase == 'test':
        seg.test()

if __name__ == '__main__':
    main(sys.argv)