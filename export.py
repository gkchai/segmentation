# author: kcgarikipati@gmail.com

# This is the main script for exporting models to be used under Flask/REST or TF-serving.
# As of 11/01/2018, the export framework in tensorflow is still under development and two main
# approaches are available: 1) "freeze_graph" is the commonly used approach for Flask/REST service and
# generates a single ".pb" file; 2) "saved_model" is the other method used for exporting ".pb" file
# and variable information file that can be served using tf-serving.
#
# Discussions on tensorflow exporting are available online. For e.g.,
# https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/i1USytvyMgk
# https://reddit.com/r/MachineLearning/comments/6nupas/p_tensorflow_whats_the_best_way_to_saverestore/
# https://github.com/Vetal1977/tf_serving_example/blob/master/svnh_semi_supervised_model_saved.py

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import time
import loader
import cv2
from collections import namedtuple
import settings
from utils.cv import draw_mask_layer, draw_mask_contour, image_to_array, image_to_string
from utils.settings import SettingsManager
import sys
from utils.general import dict_to_obj, dataset_to_path, merge_dicts, get_logger, silence_tf_logger
import shutil
import model
import pdb
import pandas as pd
import json
import traceback

# silence TF messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
silence_tf_logger()
logger = get_logger('export')


class TestClass:
    def __init__(self, graph, sess, target='saved_model'):
        self.test_graph = graph
        self.sess = sess
        self.target = target

    @staticmethod
    def create_dataset(cfg, image_data):
        if image_data:
            imgs = image_to_array(image_data)
            imgs_str = image_to_string(image_data)
            masks = []
            names = image_data
        else:
            files_csv = dataset_to_path(cfg.test_data,
                            os.path.join(cfg.data_dir, 'test'), cfg.dim)
            imgs_str = []
            for file_csv in files_csv:
                df = pd.read_csv(file_csv, header=None)
                img_paths = df[0]
                imgs_str.extend(image_to_string(img_paths))

            l = loader.Loader(cfg)
            imgs, masks, names = l.create_eval_data(dataset_to_path(cfg.test_data,
                                os.path.join(cfg.data_dir, 'test'), cfg.dim))

        dataset = namedtuple('data_collection', ['images', 'images_str' 'masks', 'names'])
        dataset.images = imgs
        dataset.images_str = imgs_str
        dataset.masks = masks
        dataset.names = names

        return dataset

    def run(self, tensor_names_dict, data, output_path):

        # get necessary tensors by name
        input_tensor = self.test_graph.get_tensor_by_name(
            tensor_names_dict['input_tensor'])
        mask_tensor = self.test_graph.get_tensor_by_name(
            tensor_names_dict['mask_tensor'])
        score_tensor = self.test_graph.get_tensor_by_name(
            tensor_names_dict['score_tensor'])
        if self.target == 'saved_model':
            is_train_tensor = tf.constant(False)
        else:
            is_train_tensor = self.test_graph.get_tensor_by_name(
                                tensor_names_dict['is_train_tensor'])

        # make prediction
        elapsed_time_sum = 0
        for idx in range(len(data.images)):

            start_time = time.time()

            # feed each image and run the session
            if self.target == 'saved_model':
                mask_predict, score = self.sess.run([mask_tensor, score_tensor],
                                            feed_dict={input_tensor: [data.images_str[idx]],
                                            is_train_tensor: False})
            else:
                mask_predict, score = self.sess.run([mask_tensor, score_tensor],
                                        feed_dict={input_tensor:[data.images[idx]],
                                        is_train_tensor: False})

            # dont count the first measurement
            if idx > 0:
                elapsed_time_sum += time.time() - start_time

            img_gnd = cv2.cvtColor(data.images[idx] * 255, cv2.COLOR_RGB2BGR)

            # scale, int and threshold mask prediction
            mask_predict = mask_predict * 255
            mask_predict = np.squeeze(mask_predict.astype(np.uint8))
            _, mask_predict = cv2.threshold(mask_predict, 127, 255, cv2.THRESH_BINARY)

            if len(data.masks):
                mask_gnd = np.squeeze((data.masks[idx] * 255).astype(np.uint8))
                # print("performance = {}".format(np.sum(mask_gnd==mask_predict)/(np.prod(mask_gnd.shape))))

                mask_gnd = cv2.cvtColor(mask_gnd, cv2.COLOR_GRAY2BGR)
                img_w_mask_gnd = draw_mask_contour(mask_gnd, mask_predict, (255, 0, 0))
                mask_w_contour = cv2.resize(img_w_mask_gnd, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(output_path, 'contour_' + data.names[idx]), mask_w_contour)

            img_w_mask_predict = draw_mask_layer(img_gnd, mask_predict, (0, 255, 0), alpha=0.3)
            cv2.putText(img_w_mask_predict, 's={0:.4f}'.format(score), (180, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            img_w_layer = cv2.resize(img_w_mask_predict, (256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_path, 'layer_' + data.names[idx]), img_w_layer)
            # logger.info("{0}: {1:.4f}".format(data.names[idx], score))

        logger.info("Mean inference-time per Image = {0:.2f}ms".format(elapsed_time_sum * 1000 /(len(data.images)-1)))


def load_graph_from_pb(pb_filename):
    """load the protobuf file from the disk and parse it to retrieve the unserialized graph_def"""

    with tf.gfile.GFile(pb_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def export_freeze_graph(export_path, export_model_name, input_ckpt_path, input_graph_name, latest_ckpt_filename=None):
    """Generate two models (frozen and optimized) for export from meta graph and checkpoint
       Uses freeze graph method for exporting
    """

    input_graph_path = os.path.join(export_path, input_graph_name + '.pbtxt')
    checkpoint_path = tf.train.latest_checkpoint(input_ckpt_path, latest_filename=latest_ckpt_filename)

    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output/y_pred, output/score"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = os.path.join(export_path,'frozen_' + export_model_name + '.pb')
    output_optimized_graph_name = os.path.join(export_path,'optimized_' + export_model_name + '.pb')
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input/x", "input/is_train"],  # an array of the input node(s)
        ["output/y_pred", "output/score"],  # an array of output nodes
        [tf.float32.as_datatype_enum, tf.bool.as_datatype_enum])
    #
    # # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

    # copy info.json file
    shutil.copy2(os.path.join(input_ckpt_path, 'info.json'), os.path.join(export_path, 'info.json'))
    logger.info("Successfully exported model to {}".format(export_path))


# https://stackoverflow.com/questions/48028511/connect-input-and-output-tensors-of-a-pre-trained-graph-and-the-current-graph-in
def export_saved_model(cfg, export_path, export_model_name, input_ckpt_path, latest_ckpt_filename=None):
    """ Generate models for export from graph and checkpoint.
        Uses savedModel method for exporting required for TF-serving
    """

    cfg_loader = loader.Loader(cfg)

    # clear all graphs
    tf.reset_default_graph()

    with tf.Graph().as_default():

        checkpoint_path = tf.train.latest_checkpoint(input_ckpt_path, latest_filename=latest_ckpt_filename)

        # Inject placeholder into the graph for the serialized tf.Example proto input
        # example proto is mandatory serialization used by tf-serving
        serialized_tf_example = tf.placeholder(tf.string, name='input_image')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        # maps proto def to tensor dicts
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)

        # jpeg = <tf.Tensor 'ParseExample/ParseExample:0' shape=(?,) dtype=string>
        jpeg = tf_example['image/encoded']

        def preprocess_image(image_buffer):
            image = cfg_loader.preprocess(image_buffer, 'jpeg', 3)
            return image

        image = tf.map_fn(preprocess_image, jpeg, dtype=tf.float32)

        # using the model class is a temporary fix to make export_saved_model work.
        # ideally you should be able to convert stored checkpoint and meta files directly
        # to saved_model. Instead here we recreate the input and output tensors
        _model = model.Model(cfg)
        logits_pred = _model.create_net(image, training=False)
        [y_pred_soft, y_pred, score] = _model.create_eval_ops(logits_pred)

        imported_saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 1})) as sess:

            # all saved variables from checkpoint are restored
            imported_saver.restore(sess, checkpoint_path)

            export_serving_path = os.path.join(export_path, 'serving')
            if os.path.exists(export_serving_path):
                shutil.rmtree(export_serving_path)

            # create model builder
            builder = tf.saved_model.builder.SavedModelBuilder(export_serving_path)

            # create tensors info
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(jpeg)
            predict_tensor_y_pred_info = tf.saved_model.utils.build_tensor_info(y_pred)
            predict_tensor_score_info = tf.saved_model.utils.build_tensor_info(score)

            # build prediction signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': predict_tensor_inputs_info},
                    outputs={'score': predict_tensor_score_info,
                             'mask': predict_tensor_y_pred_info,
                             },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            # save the model
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images': prediction_signature
                },
                legacy_init_op=legacy_init_op)

            builder.save()
        shutil.copy2(os.path.join(input_ckpt_path, 'info.json'), os.path.join(export_serving_path, 'info.json'))
        logger.info("Successfully exported model to {}".format(export_serving_path))


def export_test_saved_model(cfg, image_paths=None):
    """test exported model using saved_model format. image_data
    is list of image files on which preciction is run. if
    no image_data is provided, uses loader test dataset for prediction
    """

    model_dir = os.path.join(cfg.export_dir, cfg.model_id, 'serving')
    logger.info("\n.......Testing %s ........... \n" % model_dir)

    my_config = tf.ConfigProto(device_count={"GPU": 1})
    my_config.gpu_options.allow_growth = False

    OUTPUT_PATH = os.path.join(cfg.out_dir, cfg.model_id)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # We launch a Session to test the exported file
    with tf.Session(config=my_config, graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        # logger.info(model)

        test_suite = TestClass(tf.get_default_graph(), sess, 'saved_model')
        dataset = test_suite.create_dataset(cfg, image_paths)

        test_suite.run(
            {"input_tensor": model.signature_def['predict_images'].inputs['images'].name,
             "mask_tensor": model.signature_def['predict_images'].outputs['mask'].name,
             "score_tensor": model.signature_def['predict_images'].outputs['score'].name},
            dataset, OUTPUT_PATH
        )


def export_test_freeze_graph(cfg, image_paths=None, model_name='frozen_model'):
    """test exported model using freeze_grpah format. image_data
    is list of image files on which preciction is run. if
    no image_data is provided, uses loader test dataset for prediction
    """

    model_filename = os.path.join(cfg.export_dir, cfg.model_id, '{}.pb'.format(model_name))
    logger.info("\n.......Testing %s ........... \n" % model_filename)

    # We use our "load_graph" function
    graph = load_graph_from_pb(model_filename)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     logger.info(op.name)

    OUTPUT_PATH = os.path.join(cfg.out_dir, cfg.model_id)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    my_config = tf.ConfigProto(device_count={"GPU": 0})
    my_config.gpu_options.allow_growth = False

    # We launch a Session to test the exported file
    with tf.Session(config=my_config, graph=graph) as sess:

        test_suite = TestClass(graph, sess, 'freeze_graph')
        dataset = test_suite.create_dataset(cfg, image_paths)
        test_suite.run(
            {"input_tensor": 'prefix/input/x:0', "is_train_tensor": "prefix/input/is_train:0",
             "mask_tensor": "prefix/output/y_pred:0", "score_tensor": "prefix/output/score:0"},
            dataset, OUTPUT_PATH
        )


if __name__ == '__main__':

    base_cfg = settings.Config()
    sm = SettingsManager.get(base_cfg.get_options(), base_cfg.__dict__)
    parser = argparse.ArgumentParser(description="Export script")
    parser.add_argument('--model_id', dest='model_id', help='model_id for reading saved checkpoint', required=True)
    parser.add_argument('--target', dest='target', choices=['freeze_graph', 'saved_model'],
                        default='freeze_graph',
                        help='target for export: freeze_graph for REST API, saved_model for TF-Serving ',
                        required=True)

    args = parser.parse_args()
    model_id = args.model_id
    target = args.target

    sm.update({'model_id': model_id})
    ckpt_dir = os.path.join(base_cfg.ck_dir, model_id)
    if not os.path.exists(ckpt_dir):
        logger.info("Provided checkpoint dir {} does not exist".format(ckpt_dir))
        sys.exit(0)

    json_dict = json.loads(open(os.path.join(ckpt_dir, 'info.json')).read())
    stored_dict = merge_dicts(json_dict['params'], json_dict['dataset'])
    sm.update(stored_dict)

    cfg = dict_to_obj(sm.__dict__())

    model_basename = 'seg_model'
    try:
        if target == 'freeze_graph':
            export_freeze_graph(export_path=os.path.join(cfg.export_dir, model_id), export_model_name=model_basename,
                                input_ckpt_path=os.path.join(cfg.ck_dir, model_id), input_graph_name =model_basename,
                                latest_ckpt_filename='best_checkpoint')
            # test the exported model
            export_test_freeze_graph(cfg, model_name='frozen_{}'.format(model_basename))

        else:
            export_saved_model(cfg, export_path=os.path.join(cfg.export_dir, model_id), export_model_name=model_basename,
                               input_ckpt_path=os.path.join(cfg.ck_dir, model_id), latest_ckpt_filename='best_checkpoint'
                               )
            export_test_saved_model(cfg)

    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)