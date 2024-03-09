import sys
from os import path


def generate_array(src, name, fout):
    fout.write(b'const unsigned char __%s[] = {' % name.encode('utf-8'))
    with open(src, 'rb') as fin:
        for b in fin.read():
            fout.write(b'0x%02X,' % b)
    fout.write(b'};\n')
    def_code = 'struct __spv_code __%s_code = {.pcode = __%s, .size = sizeof(__%s)};\n' % (name, name, name)
    fout.write(def_code.encode('utf-8'))


if __name__ == "__main__":
    names = []
    cpp_name = sys.argv[1]
    header_name = sys.argv[2]
    header = path.basename(header_name)
    header_gurad = f'__{header}__'.upper().replace('.', '_')

    with open(cpp_name, 'wb+') as fout:
        fout.write(f'#include "{header}"\n'.encode('utf-8'))
        for f in sys.argv[3:]:
            name = path.basename(f).replace('.', '_')
            names.append(name)
            print(f'write {f} to array {name} in file {cpp_name}')
            generate_array(f, name, fout)

    with open(header_name, 'wb+') as fout:
        fout.write(f'#ifndef {header_gurad}\n#define {header_gurad}\n'.encode('utf-8'))
        fout.write("#include <cstddef>\nstruct __spv_code{const unsigned char* pcode; const size_t size;};\n".encode('utf-8'))
        for name in names:
            fout.write(f'extern struct __spv_code __{name}_code;\n'.encode('utf-8'))
        fout.write(b'#endif')
