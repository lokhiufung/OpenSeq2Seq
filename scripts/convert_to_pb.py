import tensorflow as tf

from open_seq2seq.utils.utils import get_base_config, check_logdir, create_model

# Change with your configs here
args_S2T = ["--config_file=infer_S2T/ds2_medium.py",
        "--mode=interactive_infer",
        "--logdir=experiments/ds2_medium_man-700/",
        "--batch_size_per_gpu=10",
]

def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None, checkpoint=checkpoint)
    return model, checkpoint

def convert_to_pb():
    model_S2T, checkpoint_S2T = get_model(args_S2T, "S2T")

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
    vars_S2T = {}
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        if "S2T" in v.name:
            vars_S2T["/".join(v.op.name.split("/")[1:])] = v
    saver_S2T = tf.train.Saver(vars_S2T)
    saver_S2T.restore(sess, checkpoint_S2T)

    input_tensors = model_S2T.get_data_layer(0).input_tensors
    loss, outputs = model_S2T.build_trt_forward_pass_graph(
                input_tensors,
                gpu_id=0,
                checkpoint=checkpoint_S2T)    

    output_node_names = ["ForwardPass/fully_connected_ctc_decoder/logits"]

    # fix batch norm nodes
    # get graph definition
    gd = sess.graph.as_graph_def()

    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    # fix

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    with open('tmp/output_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

if __name__ == '__main__':
    convert_to_pb()
    