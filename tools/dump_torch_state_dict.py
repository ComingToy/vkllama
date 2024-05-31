import torch
import sys
import os
from models.proto import llama2_model_pb2 as llama2
import functools

def __add_variable(variables, state_dict, name):
        v = state_dict[name]
        variable = variables.variables.add()
        variable.name = name
        variable.shape.extend(v.shape)
        variable.dtype = llama2.FLOAT16
        variable.data = v.reshape(-1).numpy().tobytes()

def main():
    if len(sys.argv) != 3:
        print(f'usage: sys.argv[0] <input path> <output path>')
        return -1

    path = sys.argv[1]
    out = sys.argv[2]
    state_dict = torch.load(path, weights_only=True)

    # input layer
    variables = llama2.Variables()
    add_variable = functools.partial(__add_variable, variables=variables, state_dict=state_dict)

    def write_to_file(pb, fname):
        with open(os.path.join(out, fname), 'wb+') as fout:
            fout.write(pb.SerializeToString())

    add_variable(name='model.embed_tokens.weight')
    print(f'add input layer')
    write_to_file(variables, 'input_layer.pb')
    print(f'dump input layer')
    
    for i in range(26):
        variables = llama2.Variables()
        add_variable = functools.partial(__add_variable, variables=variables, state_dict=state_dict)

        add_variable(name=f'model.layers.{i}.input_layernorm.weight')
        add_variable(name=f'model.layers.{i}.self_attn.k_proj.weight')
        add_variable(name=f'model.layers.{i}.self_attn.q_proj.weight')
        add_variable(name=f'model.layers.{i}.self_attn.v_proj.weight')
        add_variable(name=f'model.layers.{i}.self_attn.o_proj.weight')
        add_variable(name=f'model.layers.{i}.mlp.gate_proj.weight')
        add_variable(name=f'model.layers.{i}.mlp.down_proj.weight')
        add_variable(name=f'model.layers.{i}.mlp.up_proj.weight')
        add_variable(name=f'model.layers.{i}.post_attention_layernorm.weight')

        print(f'dump block{i}')
        write_to_file(variables, f'block{i}.pb')

    variables = llama2.Variables()
    add_variable = functools.partial(__add_variable, variables=variables, state_dict=state_dict)
    add_variable(name='model.norm.weight')
    add_variable(name='lm_head.weight')
    write_to_file(variables, 'output_layer.pb')
    print(f'dump output layer')
    return 0


if __name__ == "__main__":
    main() 
