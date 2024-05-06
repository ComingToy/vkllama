import tensorflow as tf
import sys
tf.compat.v1.disable_eager_execution()

def parse_block(block_idx, constants):
    #parse attn  
    block_suffix = f'_{block_idx}' if block_idx > 0 else ''
    ff_layer_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/feed_forward{block_suffix}/ff_w%d/Tensordot/ReadVariableOp'
    feed_forward_layer = {'w1': constants[ff_layer_fmt % 1], 'w2': constants[ff_layer_fmt % 2], 'w3': constants[ff_layer_fmt % 3]}

    transformer_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/gq_attention{block_suffix}/%s/Tensordot/ReadVariableOp'
    transformer_layer = {'Wk': constants[transformer_fmt % 'Wk'], 'Wq': constants[transformer_fmt % 'Wq'], 'Wv': constants[transformer_fmt % 'Wv'], 'Wo': constants[transformer_fmt % 'Wo']}

    rms_norm_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/rms_norm_{block_idx + 1}/mul_1/ReadVariableOp'
    rms_norm_layer = {'weight': constants[rms_norm_fmt]}

    return {'rms_norm': rms_norm_layer, 'transformer': transformer_layer, 'feed_forward': feed_forward_layer}

def parse_input(constants):
    embeddings = constants['llama2_model/26387']
    rms_norm_layer = {'weight': constants['llama2_model/StatefulPartitionedCall/rms_norm/mul_1/ReadVariableOp']}

    return {'embeddings': embeddings, 'rms_norm': rms_norm_layer}


def dump_constants(path, inputs):
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        graphdef = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path, 'rb') as fin:
            graphdef.ParseFromString(fin.read())
        tf.import_graph_def(graphdef, name='')

    nodes = [n for n in graph.as_graph_def().node if n.op == 'Const']
    names = [n.name for n in nodes]
    tensors = [graph.get_operation_by_name(n.name).outputs[0] for n in nodes]

    sess = tf.compat.v1.InteractiveSession(graph=graph)
    constants = {name: sess.run(tensor) for name, tensor in zip(names, tensors)}

    blocks = [parse_block(i, constants) for i in range(6)]
    input_layer = parse_input(constants)
    for k, v in constants.items():
        print(f'save {k} into numpy file, shape = {v.shape}')

if __name__ == "__main__":
    chkpt = sys.argv[1]
    dump_constants(chkpt, '')
