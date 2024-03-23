import sys
from os import path
import jinja2

header_tmpl = """
#ifndef {{ header_guard }}
#define {{ header_guard }}
#include <cstddef>
#include <stdint.h>
{% for spv_name in spv_names %}
extern const uint8_t* __get_{{ spv_name }}_code();
extern size_t __get_{{ spv_name }}_size();
{% endfor %}
#endif
"""

source_tmpl = """
#include <cstddef>
#include <stdint.h>
{% for spv_name, binary in frags %}
static uint8_t __{{ spv_name }}_code[] = { {{ binary }} };
const uint8_t* __get_{{ spv_name }}_code() { return __{{ spv_name }}_code; }
size_t __get_{{ spv_name }}_size() { return sizeof(__{{ spv_name }}_code); }
{% endfor %}
"""

def generate_array(src, name):
    bins = b''
    with open(src, 'rb') as fin:
        for b in fin.read():
            bins += (b'0x%02X, ' % b)
    return bins

if __name__ == "__main__":
    names = []
    cpp_name = sys.argv[1]
    header_name = sys.argv[2]
    header = path.basename(header_name)
    header_guard = f'__{header}__'.upper().replace('.', '_')

    spv_names = []
    for spv in sys.argv[3:]:
        name = path.basename(spv).replace('.', '_')
        spv_names.append(name)

    tmpl = jinja2.Template(header_tmpl)
    header_code = tmpl.render(header_guard=header_guard, spv_names=spv_names)
    with open(header_name, 'w+', encoding='utf-8') as fout:
        fout.write(header_code)

    frags = []
    for src, name in zip(sys.argv[3:], spv_names):
        frags.append((name, generate_array(src, name)))

    tmpl = jinja2.Template(source_tmpl)
    cpp_code = tmpl.render(frags=frags).replace("b'", '').replace("'", '')
    with open(cpp_name, 'w+', encoding='utf-8') as fout:
        fout.write(cpp_code)
