import tensorflow as tf
import sys
from src.proto import llama2_model_pb2 as llama2
# from tools import llama2_model_pb2 as llama2
tf.compat.v1.disable_eager_execution()

def parse_block(block_idx, constants):
    #parse attn  
    variables = []
    block_suffix = f'_{block_idx}' if block_idx > 0 else ''
    ff_layer_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/feed_forward{block_suffix}/ff_w%d/Tensordot/ReadVariableOp'
    w1, w2, w3 = constants[ff_layer_fmt % 1], constants[ff_layer_fmt % 2], constants[ff_layer_fmt % 3]
    variables.append(llama2.Variable(name='block_{block_idx}/feed_forward/w1', shape=w1.shape, f32_values=w1.reshape(-1)))
    variables.append(llama2.Variable(name='block_{block_idx}/feed_forward/w2', shape=w2.shape, f32_values=w2.reshape(-1)))
    variables.append(llama2.Variable(name='block_{block_idx}/feed_forward/w3', shape=w3.shape, f32_values=w3.reshape(-1)))

    transformer_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/gq_attention{block_suffix}/%s/Tensordot/ReadVariableOp'
    Wk = constants[transformer_fmt % 'Wk']
    Wq = constants[transformer_fmt % 'Wq']
    Wv = constants[transformer_fmt % 'Wv']
    Wks = []
    Wqs = []
    Wvs = []
    for i in range(8):
        wk = Wk[:, i*512:(i+1)*512]
        wq = Wq[:, i*512:(i+1)*512]
        wv = Wv[:, i*512:(i+1)*512]
        variables.append(llama2.Variable(name=f'block_{block_idx}/Wk/head_{i}', shape=wk.shape, f32_values=wk.reshape(-1)))
        variables.append(llama2.Variable(name=f'block_{block_idx}/Wq/head_{i}', shape=wq.shape, f32_values=wq.reshape(-1)))
        variables.append(llama2.Variable(name=f'block_{block_idx}/Wv/head_{i}', shape=wv.shape, f32_values=wv.reshape(-1)))

    rms_norm_fmt = f'llama2_model/StatefulPartitionedCall/llama2_block{block_suffix}/rms_norm_{block_idx + 1}/mul_1/ReadVariableOp'
    weight = constants[rms_norm_fmt]
    variables.append(llama2.Variable(name=f'block_{block_idx}/rms_norm/weight', shape=weight.shape, f32_values=weight.reshape(-1)))
    return variables


def parse_input(constants):
    embeddings = constants['llama2_model/26387']
    weight = constants['llama2_model/StatefulPartitionedCall/rms_norm/mul_1/ReadVariableOp']
    return [llama2.Variable(name=f'input/embeddings', shape=embeddings.shape, f32_values=embeddings.reshape(-1)), llama2.Variable(name='input/rms_norm/weight', shape=weight.shape, f32_values=weight.reshape(-1))]


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
    for k, v in constants.items():
        print(f'save {k} into numpy file, shape = {v.shape}')

    variables = []
    variables += parse_input(constants)
    for i in range(6):
        variables += parse_block(i, constants)

    variables = llama2.Variables(variables=variables)

    with open('variables.pb', 'wb+') as fout:
        fout.write(variables.SerializeToString())

if __name__ == "__main__":
    chkpt = sys.argv[1]
    dump_constants(chkpt, '')
