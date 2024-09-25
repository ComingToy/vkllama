#define u8BufToU32(buf, idx)                                                  \
  ((uint (buf[idx + 3]) << 24) | (uint (buf[idx + 2]) << 16)                  \
   | (uint (buf[idx + 1]) << 8) | (uint (buf[idx])))
